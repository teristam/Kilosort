#%%
%reload_ext autoreload
%autoreload 2
from kilosort.io import *
import kilosort 
from kilosort.run_kilosort import get_run_parameters
import torch
from kilosort import preprocessing
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from matplotlib import gridspec
from tqdm.auto import tqdm
from icecream import ic

#%%

session_id = 'RE012_2024-02-21_14-13-08_VR_bar_final'
data_dir = Path(f'/home/MRC.OX.AC.UK/ndcn1330/ettin/Julien/Data/head-fixed/neuropixels/{session_id}/Record Node 101/experiment1/recording2/continuous/Neuropix-PXI-100.ProbeA-AP') 


device = torch.device('cuda:0')
settings = {'data_dir': data_dir, 
            'n_chan_bin': 384, 
            'device':device,
            'tmax':100,
            # 'artifact_threshold': 1000,
            'remove_artifact_spikes':True,
            'batch_size': 30000*8,
           'results_dir': 'test_output2'}

ops, st_orig, clu_orig, tF, Wall, similar_templates, is_ref, est_contam_rate = \
    kilosort.run_kilosort(settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat',
                         save_extra_vars=True)

# some st may contains duplicates, they will be removed when saved to disk

torch.cuda.empty_cache()
# %%
