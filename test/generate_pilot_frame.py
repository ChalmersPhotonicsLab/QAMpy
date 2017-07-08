# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:50:12 2017

@author: fotonik
"""

from scipy.io import loadmat, savemat
import numpy as np
from dsp import signals, equalisation, modulation, utils, phaserecovery, dsp_cython, signal_quality, pilotbased_transmitter
from dsp.prbs import make_prbs_extXOR

""" 
Testing the transmitter
"""
# Tx Config
M = 64
os = 2
# Pilot Settings

# Total frame length
frame_length = 2**18
# Initial number of pilot tones for equalizer pre-convergence
pilot_seq_len = 256
# Repetative pilot symbols for phase tracking and equalizer update
pilot_ins_ratio = 32 


# Settings if PRBS is seleted for generation
PRBS = False
PRBSorder=15,
PRBSseed=None


frame_symbs, data_symbs, pilot_symbs = pilotbased_transmitter.gen_dataframe_with_phasepilots(M,1,frame_length = frame_length, pilot_seq_len = pilot_seq_len)
savemat('pilot_pattern_2e16_64qam.mat',{'frame_symbs':frame_symbs, 'data_symbs':data_symbs, 'pilot_symbs':pilot_symbs})


#frame_symbs_X = np.roll(frame_symbs, 6523)
#frame_symbs_Y = np.roll(frame_symbs, 6523)

#frame_symbs = np.vstack([frame_symbs_X,frame_symbs_Y])
#pilot_symbs = np.vstack([pilot_symbs,pilot_symbs])


#tx_sig = sim_tx(frame_symbs, os,snr = 50, linewidth = None, freqoff = None, rot_angle=None)



# Plot Result
#plt.figure()
#plt.semilogy(np.fft.fftshift(np.fft.fftfreq(len(tx_sig[0,:]),1/os)), np.fft.fftshift(np.abs(np.fft.fft(tx_sig[0,:]))**2))
#
#plt.figure()
#plt.plot([symbol_seq[:100].real,symbol_seq[:100].imag],'.')






