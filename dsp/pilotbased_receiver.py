#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:11:02 2017

@author: mazurm
"""

import numpy as np
from dsp import signals, equalisation, modulation, utils, phaserecovery, dsp_cython, signal_quality, ber_functions
from scipy.io import loadmat, savemat
import matplotlib.pylab as plt

# Tx Config
M = 32
os = 2
symb_rate = 20e9
snr = None #dB
linewidth = None # Linewidth symbol-rate product
fo = None # Frequency offset MHz

# Pilot Settings

# Total frame length
frame_length = 2**18
# Initial number of pilot tones for equalizer pre-convergence
pilot_seq_len = 256
# Repetative pilot symbols for phase tracking and equalizer update
pilot_ins_ratio = 32 
# Settings if PRBS is seleted for generation
PRBS = False

# Var names from Tx

#frame_symbs, data_symbs, pilot_symbs = gen_dataframe_withpilots(128)

#tx_sig = sim_tx(frame_symbs, os)

"""
 
Locate pilot sequence

"""

step_size = pilot_seq_len / os
num_steps = frame_length / step_size


tx_sig_shift = np.roll(tx_sig,int(1e4))

numTests = frame_length/pilot_seq_len

batchVar = np.zeros([numTests,1])


mu = 1e-3
M_pilot = 4
Ntaps = 25
adap_step = True

# Find suitable starting point
for i in np.arange(0,numTests):
    wx, err = equalisation.equalise_signal(tx_sig_shift[(i)*pilot_seq_len:(i+1)*pilot_seq_len], os, mu, M_pilot, Ntaps = Ntaps, Niter=10, method="cma",adaptive_stepsize = adap_step)
    batchVar[i] = np.var(err[0,-100:])


minIndex = np.argmin(batchVar)
if batchVar[minIndex+1] > batchVar[minIndex-1]:
    ud_ind_search = -1
    tmpSig = tx_sig_shift[(minIndex-1-0.5)*pilot_seq_len:(minIndex+1+1)*pilot_seq_len]
    
else:
     ud_ind_search = 1   
     tmpSig = tx_sig_shift[(minIndex-0.5)*pilot_seq_len:(minIndex+2+1)*pilot_seq_len]

# Equalize the optimal part! 
wx, err = equalisation.equalise_signal(tmpSig, os, mu, M_pilot, Ntaps = Ntaps, Niter=10, method="cma",adaptive_stepsize = adap_step)
symbs_out = equalisation.apply_filter(tmpSig,os,wx)

plt.plot(symbs_out[0,70:-70].real,symbs_out[0,70:-70].imag,'.')

# Reference pilot sequence 
ref_symbs = pilot_symbs[0:pilot_seq_len]

# Align received symbols to 
xcorr = np.correlate(np.angle(symbs_out[0,:]),np.angle(ref_symbs),mode='full')
symb_delay = np.argmax(xcorr)
test_alignment = symbs_out[0,symb_delay-pilot_seq_len +1 :symb_delay+1]




testSig = tx_sig_shift[(minIndex-1-0.5)*pilot_seq_len+os*(symb_delay-pilot_seq_len+1):(minIndex-0.5)*pilot_seq_len+os*(symb_delay-pilot_seq_len+1)+pilot_seq_len*os+Ntaps+10]
wx, err = equalisation.equalise_signal(testSig, os, mu, M_pilot, Ntaps = Ntaps, Niter=10, method="cma",adaptive_stepsize = adap_step)
symbs_align_test = equalisation.apply_filter(testSig,os,wx)




plt.figure()
plt.plot(symbs_out[0,pilot_seq_len:].real,symbs_out[0,pilot_seq_len:].imag,'.')
plt.title("Equalized received symbols, including phase pilots")
plt.figure()
plt.plot(symbs_out[0,0:pilot_seq_len-5].real,symbs_out[0,0:pilot_seq_len-5].imag,'.')
plt.title("Equalized Pilots")
plt.figure()
plt.plot(np.abs(err[0]))
plt.tite("CMA Error")