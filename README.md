# About

The goal of this project is to detect Cognitive Fatigue onset or unique pattern using Multifractal Detrended Analysis on a fractional dynamics system data (EEG).

# MFDFA Utilization

Proper utilization of MFDFA function.

## Function Utilization

Three step MDFA utilization process on biomedical time-series:
1. Ensure time series has noise-like structure via Monofractal DFA (if noise, convert to random walk)
2. Local fluctuations close to zero should be removed within MDFA
3. Check for scale invariance, if not present for whole range modify with MDFA1. Use MDFA2 if the time instant for structural change is of importance or if there are less than ~5000 samples. 
   1. Small scaling ranges (such as scale = [7,9,11,13,15,17]) should be used in MFDFA2. Note that due to the small scale used in MFDFA2, a less precise estimation of the local fluctuation RMS occurs.

## Input for Multifractal DFA

Table for proper multifractal input determined from monofractal Hurst result:

| Hurst   | Conversion Formula         | Adjustment of Hq & Ht |
| ------- | -------------------------- | --------------------- |
| <0.2    | Signal = cumsum(x-mean(x)) | -1                    |
| 0.2-0.8 | N/A                        | 0                     |
| 0.8-1.2 | N/A                        | 0                     |
| 1.2-1.8 | Signal = diff(x)           | +1                    |
| >1.8    | Signal = diff(diff(x))     | +2                    |

*table from : Introduction to Multifractal Detrended Fluctuation Analysis in Matlab*

# TODO

- [ ] Implement MFDFA2 or determine if it shouldn't be utilized (likely the case).
  - [ ] Possibly implement noise2rw, want to test data first though and determine if we need it.