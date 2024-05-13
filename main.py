import numpy as np
import os
import mne
import warnings
warnings.filterwarnings("ignore")

# %load_ext autoreload
# %autoreload 2
from MDFA import MDFA

def test1(dataset_root, files, scales, q):
  for fname in files:
    try:
      vhdr_file_path = os.path.join(dataset_root,f'{fname}/{fname}_D1.vhdr')
      raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)
      signal, times = raw[:]
      mdfa = MDFA(signal,'A2', scales, q, True, 2, k=False)
      mdfa.MDFA_whole()
    except:
      print("Skipped: "+str(fname))

def test2(dataset_root, files, scales, q):
  for fname in files:
    try:
      vhdr_file_path = os.path.join(dataset_root,f'{fname}/{fname}_D1.vhdr')
      raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)
      signal, times = raw[:]
      mdfa = MDFA(signal,'A2', scales, q, False, 2, k=100000)
      mdfa.MDFA_segments()
    except Exception as e:
      print("Skipped: "+str(fname)+", Error: "+str(e))

if __name__ == '__main__':

  # Root folder
  NASA_dataset_root = '../NASA-EEG-DATA/EEG'

  # Get files
  files = os.listdir(NASA_dataset_root)
  filtered_files = [file for file in files if 'BB' in file]
  filtered_files,len(filtered_files)

  # Function variables
  scales = [16, 32, 64, 128, 256, 512, 1024]
  q = np.linspace(-10, 10, 100)

  test1(NASA_dataset_root, filtered_files, scales, q)
  test2(NASA_dataset_root, filtered_files, scales, q)