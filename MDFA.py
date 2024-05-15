import numpy as np
import matplotlib.pyplot as plt


class MDFA:
  def __init__(self,signal, channel, scales, q, plot,order,k,fname):
    self.order=order
    self.q =q
    self.scales = scales
    self.k = k
    self.fname = fname
    self.channel = channel
    if channel == 'A2': channel = 0
    elif channel == 'Cz': channel = 1
    elif channel == 'EoG_R': channel = 2
    elif channel == 'EoG_L': channel = 3
    else:
      raise("channel is not in the data")

    self.data = signal[channel]
   
    if not k or k == len(self.data):
      self.MDFA_whole()
      if plot:
        self.plot_results()
    
    else:
      self.segments = self.make_segments()
      self.MDFA_segments()
      # if plot:
      #    self.plot_dq_segments()

    

  def calculate_RMS_Fq(self,input_data):
    X = np.cumsum(input_data) - np.mean(input_data)
    X = X.reshape(-1, 1)
    RMS = [[] for _ in self.scales]
    Fq = np.zeros((len(self.q), len(self.scales)))
    for ns, scale in enumerate(self.scales):
        segments = len(X) // scale
        for v in range(segments):
            idx_start = v * scale
            idx_end = (v + 1) * scale
            index = np.arange(idx_start, idx_end)
            C = np.polyfit(index, X[index], self.order)
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

  def calculate_tq_Dq_hq(self,Hq):
    tq = Hq * self.q - 1
    tq_diff = np.diff(tq)
    # Calculate hq
    delta_q = self.q[1] - self.q[0]  # The difference between the first two q values
    hq = tq_diff / delta_q
    # Calculate Dq
    Dq = (self.q[:-1] * hq) - tq[:-1]  # q[:-1] means all q elements except the last one
    return tq,Dq,hq

  def calculate_Hq_qRegLine(self,Fq):
    Hq = np.zeros(len(self.q))
    qRegLine = []

    # Perform linear regression on the log-log values
    for nq in range(len((self.q))):
        C = np.polyfit(np.log2(self.scales), np.log2(Fq[nq, :]), self.order)
        Hq[nq] = C[0]
        qRegLine.append(np.polyval(C, np.log2(self.scales)))

    return Hq,qRegLine

  def plot_results(self):
    """
    This function plots all the tq,Hq,ha,Dq results of a given data
    """
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # 1 row, 5 columns
    plot_settings = [
        (0, self.q, self.Hq, 'q-order', 'Hq','Hurst exponent'),
        (1, self.q, self.tq, 'q-order', 'tq','Mass exponent'),
        (2, self.q[1:], self.Dq, 'q-order', 'Dq','Singularity Dimension'),
        (3, self.q[1:], self.hq, 'q-order', 'hq','Singularity Exponent'),
        (4, self.hq, self.Dq, 'hq', 'Dq','Multifractal spectrum of Dq and hq')
    ]

    for index, x, y, xlabel, ylabel, lbl in plot_settings:
        axs[index].plot(x, y, 'o-',label=lbl,color='purple')
        axs[index].set_xlabel(xlabel)
        axs[index].set_ylabel(ylabel)
        axs[index].legend()

    plt.tight_layout()
    plt.savefig(f"plots/MDFA/{self.fname}_{self.channel}_all.png",dpi=300)
    plt.show()

  def MDFA_whole(self):
    """
    caclulate the Fq,RMS,Hq and other multifrcal metrics for whole signla (data). if want to run MDFA on segents or intervals of data use MDFA_segments
    """
    RMS, self.Fq = self.calculate_RMS_Fq(self.data)
    self.Hq,qRegLine = self.calculate_Hq_qRegLine(self.Fq)
    self.tq,self.Dq,self.hq = self.calculate_tq_Dq_hq(self.Hq)

  def make_segments(self):
    n = len(self.data)
    num_full_segments = n // self.k
    leftover = n % self.k
    
    segments = [self.data[i * self.k:(i + 1) * self.k] for i in range(num_full_segments)]
    
    if leftover:
        segments.append(self.data[num_full_segments * self.k:]) 

    return segments

  def make_qual_size_segments(self):
    n = len(self.data)
    original_k = self.k

    # Try to divide the data with the given k
    if n % original_k == 0:
        segments = [self.data[i * original_k:(i + 1) * original_k] for i in range(n // original_k)]
        return segments
    else:
        # Find the smallest k > original_k that divides n into equal segments
        for k in range(original_k + 1, n + 1):
            if n % k == 0:
                self.k = k  # Update k to the new value
                break

        segments = [self.data[i * self.k:(i + 1) * self.k] for i in range(n // self.k)]
        return segments
  

  def MDFA_segments(self):
    self.Fq_SGM,self.Hq_SGM , self.Dq_SGM,self.tq_SGM, self.hq_SGM = [],[],[],[],[]
    for idx, seg in enumerate(self.segments):
      RMS, self.Fq = self.calculate_RMS_Fq(seg)
      self.Hq,qRegLine = self.calculate_Hq_qRegLine(self.Fq)
      self.tq,self.Dq,self.hq = self.calculate_tq_Dq_hq(self.Hq)
      self.Hq_SGM.append(self.Hq)
      self.hq_SGM.append(self.hq)
      self.Fq_SGM.append(self.Fq)
      self.Dq_SGM.append(self.Dq)
      self.tq_SGM.append(self.tq)
      # print("Segment {} of signal :".format(idx))
      # self.plot_results()
    self.plot_dq_segments()


  def plot_dq_segments(self):
    num_plots = len(self.Dq_SGM)
    num_cols = 6
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows needed
    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))  # Adjust size as needed
    axs = axs.flatten()  # Flatten the array to simplify indexing

    # Loop over all data items and create plots
    for i, data in enumerate(self.Dq_SGM):
        axs[i].plot(self.q[1:],data)
        axs[i].set_title(f'Plot {i+1}')

    # If there are any leftover axes, turn them off
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/MDFA/{self.fname}_{self.channel}_{self.k}.png",dpi=300)
    plt.show()
