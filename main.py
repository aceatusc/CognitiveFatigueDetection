import numpy as np
import os
import mne
import warnings
warnings.filterwarnings("ignore")

# %load_ext autoreload
# %autoreload 2
from MDFA import MDFA

def test(
    whole:bool, 
    segments:bool, 
    dataset_root, 
    files:list[str], 
    scales:list[int], 
    q:np.ndarray[np.floating[np.any]], 
    plot:bool, 
    order:int, 
    k:int|bool,
    fname:str
  ) -> None:

  """
  Test dataset with MDFA

  Parameters
  --------------
  - whole:bool = Calculates the Fq, RMS, Hq and other multifractal metrics for whole signal (data).
  - segments:bool = Calculates the Fq, RMS, Hq and other multifractal metrics for segments or intervals of signal (data).
  - dataset_root = Location of dataset
  - files:list[str] = List of dataset files
  - scales:list[int] = List of scales used for MDFA
  - q:np.ndarray[np.floating[np.any]] = q-order input
  - plot:bool = Whether to display plot or not
  - order:int = aka m, use values 1-3 for smallest segments between 10-20 values
  - k:int|bool

  Displays
  --------------
  Plot (optional). Print.

  Returns
  --------------
  none
  """
  for fname in files:
    try:
      vhdr_file_path = os.path.join(dataset_root,f'{fname}/{fname}_D1.vhdr')
      raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)
      signal, times = raw[:]
      mdfa = MDFA(signal,'A2', scales, q, plot, order, k=k,fname=fname)
      if whole:
        mdfa.MDFA_whole()
      if segments:
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
  q = np.linspace(-8, 8, 100) # Testing stuff
  
  # Tests
  test(True, False, NASA_dataset_root, filtered_files, scales, q, True, 2, k=False)
  test(False, True, NASA_dataset_root, filtered_files, scales, q, False, 2, k=100000)