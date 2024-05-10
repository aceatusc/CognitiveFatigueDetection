import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne


class MDFA:
  def __init__(self,signal, channel, scales, q, plot,order):
    self.order=order
    self.q =q
    self.scales = scales

    if channel == 'A2': channel = 0
    elif channel == 'Cz': channel = 1
    elif channel == 'EoG_R': channel = 2
    elif channel == 'EoG_L': channel = 3
    else:
      raise("channel is not in the data")

    self.data = signal[channel]
    RMS, self.Fq = self.calculate_RMS_Fq()
    self.Hq,qRegLine = self.calculate_Hq_qRegLine()
    self.tq,self.Dq,self.hq = self.calculate_tq_Dq_hq()

    if plot:
        self.plot_results()

  def calculate_RMS_Fq(self):
    m = self.order

    X = np.cumsum(self.data - np.mean(self.data))
    X = X.reshape(-1, 1)
    RMS = [[] for _ in self.scales]
    Fq = np.zeros((len(self.q), len(self.scales)))
    for ns, scale in enumerate(self.scales):
        segments = len(X) // scale
        for v in range(segments):
            idx_start = v * scale
            idx_end = (v + 1) * scale
            index = np.arange(idx_start, idx_end)
            C = np.polyfit(index, X[index], m)
            fit = np.polyval(C, index)
            RMS[ns].append(np.sqrt(np.mean((X[index] - fit) ** 2)))

        RMS[ns] = np.array(RMS[ns])  # Convert segment list to array for numpy operations
        for nq, q_val in enumerate(self.q):
            if q_val != 0:
                qRMS = RMS[ns] ** q_val
                Fq[nq, ns] = np.mean(qRMS) ** (1 / q_val)
            else:
                # Special case for q = 0
                Fq[nq, ns] = np.exp(0.5 * np.mean(np.log(RMS[ns] ** 2)))
    return RMS, Fq

  def calculate_tq_Dq_hq(self):
    tq = self.Hq * self.q - 1
    tq_diff = np.diff(tq)
    # Calculate hq
    delta_q = self.q[1] - self.q[0]  # The difference between the first two q values
    hq = tq_diff / delta_q
    # Calculate Dq
    Dq = (self.q[:-1] * hq) - tq[:-1]  # q[:-1] means all q elements except the last one
    return tq,Dq,hq

  def calculate_Hq_qRegLine(self):
    Hq = np.zeros(len(self.q))
    qRegLine = []

    # Perform linear regression on the log-log values
    for nq, q_val in enumerate(self.q):
        C = np.polyfit(np.log2(self.scales), np.log2(self.Fq[nq, :]), self.order)
        Hq[nq] = C[0]
        qRegLine.append(np.polyval(C, np.log2(self.scales)))

    return Hq,qRegLine

  def plot_results(self):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # 1 row, 5 columns
    plot_settings = [
        (0, self.q, self.Hq, 'q-order', 'Hq'),
        (1, self.q, self.tq, 'q-order', 'tq'),
        (2, self.q[1:], self.Dq, 'q-order', 'Dq'),
        (3, self.q[1:], self.hq, 'q-order', 'hq'),
        (4, self.hq, self.Dq, 'hq', 'Dq')
    ]

    for index, x, y, xlabel, ylabel in plot_settings:
        axs[index].plot(x, y, 'o-')
        axs[index].set_xlabel(xlabel)
        axs[index].set_ylabel(ylabel)
        axs[index].legend()

    plt.tight_layout()
    plt.show()

