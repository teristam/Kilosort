import torch
from scipy.stats import median_abs_deviation
from matplotlib import gridspec, rcParams
from numpy.linalg import lstsq
from sklearn.cluster import MeanShift, KMeans
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import torch.nn.functional as F
from scipy.stats import median_abs_deviation as mad
from tqdm.auto import tqdm

def get_artifact_time(xabs, thres, fs, min_dist=2):
    # TOFIX: follow the one using torch
    # idx = np.nonzero(xabs>thres)[0]
    # d = np.diff(idx,axis=0,prepend=0)
    # onset = np.nonzero(d)[0] #onset of artefact
    # onset_idx = idx[onset]
    
    # remove close events
    valid_onset = []
    last_onset = onset_idx[0]
    valid_onset.append(last_onset)
    
    for i in onset_idx:
        if i - last_onset > min_dist/1000*fs:
            valid_onset.append(i)
            last_onset = i

    return valid_onset

def get_artifact_time_torch(xabs, thres, fs, min_dist=0.5):
    # idx = torch.nonzero(xabs>thres)
    above_thres = (xabs>thres)
    onset = torch.diff(above_thres, prepend=above_thres[:1])
    onset_idx = torch.nonzero(onset==1)


    # remove close events
    valid_onset = []

    if len(onset_idx)>0:
        last_onset = onset_idx[0]
        valid_onset.append(last_onset)
        for i in onset_idx[1:]:
            if i - last_onset > min_dist/1000*fs:
                valid_onset.append(i)
                last_onset = i

    return valid_onset

def collect_artifacts(valid_onset, signal, fs, window=[-1,1]):
    artifacts = np.zeros((len(valid_onset), signal.shape[0], int((window[1]-window[0])/1000*fs)))
    
    for i, idx in enumerate(valid_onset):
        start = idx+ int(window[0]/1000*fs)
        end = idx + int(window[1]/1000*fs)
        artifacts[i,:] = signal[:, start:end]

    return artifacts

def cluster_artifacts(artifact, n_clus=12):
    pca = PCA(n_components=3)
    s = pca.fit_transform(artifact[:,:]) #should I use the whole signal instead?
    
    
    clustering = KMeans(n_clusters=n_clus).fit(s)
    lbls = clustering.labels_

    return lbls

def regress_match(y, template):
    # linear scale
    A = np.vstack([template, np.ones(len(template))]).T
    m,c = lstsq(A, y, rcond=None)[0]
    return (template*m+c)

def remove_artifact(signal, artifact, clus_lbls, onset_idx):
    # artifact: a single artifact with channel x time

    win_len = artifact.shape[1]
    for i,c in enumerate(np.unique(clus_lbls)):
        idx = (clus_lbls==c)
        template = np.mean(artifact[idx,:],axis=0)
        start_idx = onset_idx-win_len//2
        end_idx = onset_idx+win_len//2

        if start_idx>0 and end_idx<signal.shape[1]:
            for j in np.nonzero(idx)[0]:
                signal[j, start_idx:end_idx] -= regress_match(signal[j,start_idx:end_idx],template)
    return signal

def strip_plot(x, window=None, fs=30000, scale=100, chan=31, ax=None):
    '''
    x should be in the shape (channel x time)
    '''
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,12))
        
    t = np.arange(x.shape[1])/fs
    if window is not None:
        xx = x[:, (t>window[0]) & (t<=window[1])]
        tt  = t[(t>window[0]) & (t<=window[1])]
    else:
        xx = x
        tt = t

    if type(chan) is int:
        chans = range(chan)
    else:
        chans = chan
    
    for i in chans:
        ax.plot(tt, xx[i,:]/scale+i)
    

def preprocess_artifact(signal, fs):

    x = signal.mean(axis=0).numpy()

    # get threshold
    xabs = np.abs(x).ravel()
    s = xabs.std()
    thres = xabs.mean() + 12*s

    valid_onset = get_artifact_time(xabs, thres, fs)

    artifacts = collect_artifacts(valid_onset, signal, fs)

    for i, onset_idx in enumerate(valid_onset):
        artifact = artifacts[i,:,:]
        lbls = cluster_artifacts(artifact)
        remove_artifact(signal, artifact, lbls, onset_idx)

def create_artifact_mask(is_artifact, kernel_size, mode='all'):
    # create a mask for artifact location in each channel
    # is_arfact should be channel x time
    x = is_artifact[None,:,:].float().to(is_artifact.device)
    chan_n = is_artifact.shape[0]
    # we need to clamp the output because multiple artifact may be in the same window
    if mode=='all':
        # mask all channels when artifact is found on any one channel
        kernel = torch.ones(1, chan_n, kernel_size).to(is_artifact.device)
        out =  1-F.conv1d(x, kernel, groups=1,padding='same').clamp(max=1)
    else:
        kernel = torch.ones(chan_n, 1, kernel_size).to(is_artifact.device)
        out =  1-F.conv1d(x, kernel, groups=chan_n,padding='same').clamp(max=1)
    return torch.squeeze(out)
        
def zero_artifact(signal, fs, window_size=1, thres=1000, mode='chan'):
    '''
    window should be in ms
    blank the region around the artifact
    Use a much faster method with convolution to create an artifact mask
    mode can be 'all' or 'chan'. For 'all', it will zero all channel, for 'chan' it will
    only zero the channel that has artifacts
    '''
    
    
    kernel_size = int(window_size/1000*fs) # window around the artifact to blank-out
    if mode=='chan':
        is_artifact = signal.abs()>thres
    else:
        d = signal.abs().mean(axis=0)>thres
        is_artifact = d.repeat(signal.shape[0],1)
        # plt.imshow(is_artifact.cpu().numpy(), aspect='auto', interpolation='none')
        # plt.xlim([30000*0.9, 30000*1.2])
        
    mask = create_artifact_mask(is_artifact,kernel_size, mode)
    # print('removing artifact')

    return signal*mask


def plot_raster(st, clu, chan_map, chan_best, t0=None,t1=None, fs = 30000, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if t0 is None:
        t0=0
    if t1 is None:
        t1 = st[-1]/fs
    
    idx2plot = np.nonzero((st>t0*fs) & (st<t1*fs))[0]
    ax.scatter(st[idx2plot]/fs, chan_best[clu[idx2plot]], s=2, color='k', marker='|')
    ymin = chan_best[clu[idx2plot]].min()
    ymax = chan_best[clu[idx2plot]].max()
    ax.set_ylim([ymax, ymin-0])
    ax.set_xlabel('time (sec.)')
    ax.set_ylabel('channel')
    ax.set_title('spikes from units')

    return ax

def draw_spike(wi, ops, cb, ax, nc=16,n_chan=383, amp=4, alpha=1, shift=0, color='k'):
    '''
    wi should be in the shape (time x channel)
    '''
    probe = ops['probe']
    xc, yc = probe['xc'], probe['yc']
    
    # location of the waveform
    ic0 = max(0, cb-nc//2)
    ic1 = min(n_chan, cb+nc//2)
    x0, y0 = xc[ic0:ic1], yc[ic0:ic1]

    idx = np.arange(ic0, ic1)+ shift
    wv = wi[:, idx]
    wv /= np.ptp(wv)
    
    for ii, (xi,yi) in enumerate(zip(x0,y0)):
        t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
        t /= wv.shape[0] / 20 #width of the contact
        ax.plot(xi + t, yi + wv[:,ii]*amp, lw=0.5, color=color,alpha=alpha)

def check_is_good(spikes, cb, ops, threshold=20):
    # spikes should be in the shape (channel, N, time)
    # detect whether a spike is a outliner
    chan2compare = ops['iCC'][:,cb]

    # find the spike template
    mean_template = spikes.mean(axis=1)

    # calculate the difference, only pay attention to surround channels
    d = (mean_template[chan2compare,None,:]-spikes[chan2compare,:,:])**2
    d = d.mean(axis=0).mean(1)

    # outliner detection using MAD
    m = mad(d)
    thres = threshold*m + np.median(m)

    is_good = (d<thres)

    return is_good

def check_is_good_pca(scores, threshold=20):
    # scores should be sample x nPC

    # find the spike template
    mean_template = scores.mean(axis=1)

    # calculate the difference, only pay attention to surround channels
    d = (mean_template[:,None]-scores)**2
    d = d.mean(axis=1)

    # outliner detection using MAD
    m = mad(d)
    thres = threshold*m + np.median(m)

    is_good = (d<thres)

    return is_good

def draw_templates(templates, ops, contam_pct, clu, thres=0.1):
    # draw the template of cluster
    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)

    probe = ops['probe']
    # x and y position of probe sites
    xc, yc = probe['xc'], probe['yc']
    nc = 16 # number of channels to show
    good_units = np.nonzero(contam_pct <= thres)[0]
    mua_units = np.nonzero(contam_pct > thres)[0]
    
    np.random.seed(0)
    
    gstr = ['good', 'mua']
    for j in range(2):
        print(f'~~~~~~~~~~~~~~ {gstr[j]} units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('unit id (num of spikes)')
        units = good_units if j==0 else mua_units 
        fig = plt.figure(figsize=(12,3), dpi=150)
        grid = gridspec.GridSpec(2,20, figure=fig, hspace=0.5, wspace=0.5)
    
        for k in range(40):
            wi = units[np.random.randint(len(units))] # unit label
            wv = templates[wi].copy()  
            cb = chan_best[wi]
            nsp = (clu==wi).sum()
            
            ax = fig.add_subplot(grid[k//20, k%20])
            n_chan = wv.shape[-1]
            ic0 = max(0, cb-nc//2)
            ic1 = min(n_chan, cb+nc//2)
            wv = wv[:, ic0:ic1]
            x0, y0 = xc[ic0:ic1], yc[ic0:ic1]
    
            amp = 4
            for ii, (xi,yi) in enumerate(zip(x0,y0)):
                t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
                t /= wv.shape[0] / 20
                ax.plot(xi + t, yi + wv[:,ii]*amp, lw=0.5, color='k')
    
            ax.set_title(f'{wi}\n({nsp})', fontsize='small')
            ax.axis('off')
        plt.show()

def extract_spikes(templates, bfile, st, clu):
    spike_width =  templates.shape[1]
    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
    nt = bfile.nt
    # cb = chan_best[clu_id]
    # nspk = 0
    
    batch_spk_idx = np.zeros((templates.shape[0],)).astype(int)
    spikes_all = [np.zeros((383, len(st[clu==i]), spike_width)) for i in range(len(batch_spk_idx))]
    
    for ibatch in tqdm(range(bfile.n_batches)):
        X, inds = bfile.padded_batch_to_torch(ibatch, return_inds=True)
        # filter the spike index for search
        if ibatch==0:
            bstart = bfile.imin
            bend = bfile.imin + bfile.NT + bfile.nt
            tstart = 0
            shift = bfile.nt
        else:
            bstart = bfile.imin + (ibatch * bfile.NT) - bfile.nt
            bend = min(bfile.imax, bstart + bfile.NT + 2*bfile.nt)
            tstart = bstart + bfile.nt
            shift = 0

        # ic(tstart, bend-bfile.nt)
        for iclu in range(len(batch_spk_idx)):
            spk_t = st[clu==iclu] #move it out
            spk_batch = spk_t[(spk_t>tstart) & (spk_t<(bend-bfile.nt))] - inds[0] - shift # need to shift by nt for some unknown reason
        
            # print(spk_batch)
            # prepare the index for faster extraction
            shift_idx = np.arange(-nt//2, nt//2)
            shift_idx = np.tile(shift_idx, (len(spk_batch),1))
            spk_idx = np.tile(spk_batch,(nt,1)).T
            spk_idx = shift_idx+spk_idx

            try:
                spikes_all[iclu][:,batch_spk_idx[iclu]:(batch_spk_idx[iclu]+len(spk_batch)),:] = X[:, spk_idx]
            except ValueError:
                print(iclu, batch_spk_idx[iclu], batch_spk_idx[iclu]+len(spk_batch), spk_idx.shape)
            batch_spk_idx[iclu] += len(spk_batch)

    return spikes_all, batch_spk_idx


def remove_artifact_spikes(clu,st, tF, threshold=10):
    
    feat = tF.reshape(tF.shape[0],-1)
    
    assert len(clu)==len(st), 'st and clu do not match'
    good_spike_mask = np.zeros_like(clu).astype(bool)
    for i in range(clu.max()+1):
        st_idx = np.nonzero(clu==i)[0]
        sel_feat = feat[clu==i,:]
        is_good_pca = check_is_good_pca(sel_feat, threshold=threshold)
        good_spike_mask[st_idx[is_good_pca]] = True

    st_valid = st[good_spike_mask,:]
    clu_valid = clu[good_spike_mask]
    tF_valid = tF[good_spike_mask]
    print(f'{(1-np.mean(good_spike_mask))*100:.2f} pct spikes removed')

    return clu_valid, st_valid, tF_valid

def get_artifact_idx(signal,thres=2000,Fs=30000):
    signal_mean = signal.abs().mean(axis=0)
    print(signal_mean.shape)
    idx = torch.nonzero(signal_mean>thres)
    d = torch.diff(idx,prepend=idx[:1],axis=0)
    #remove artifacts that are too close together
    return idx[d > 10/1000*Fs]


def calculate_rf_violation(clu,st, Fs):
    rf_violation = np.array([])
    for i in range(clu.max()+1):
        sk_time = st[clu==i]
        isi = np.diff(sk_time, prepend=0)
        rf_idx = np.nonzero(isi< 1/1000*Fs)[0]
        rf_violation = np.append(rf_violation, sk_time[rf_idx]/Fs)

    return rf_violation