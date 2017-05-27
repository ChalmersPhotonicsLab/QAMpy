#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:11:02 2017

@author: mazurm
"""

import numpy as np
from dsp import signals, equalisation, modulation, utils, phaserecovery, dsp_cython, signal_quality
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

step_size = pilot_seq_len / 2
num_steps = frame_length / step_size


mu = 1e-3
M_pilot = 4
Ntaps = 25
adap_step = False

wx, err = equalisation.equalise_signal(tx_sig[:pilot_seq_len*2], os, mu, M_pilot, Ntaps = Ntaps, Niter=10, method="cma",adaptive_stepsize = adap_step)

symbs_out = equalisation.apply_filter(tx_sig[:pilot_seq_len*100],os,wx)


plt.figure()
plt.plot(symbs_out[0,pilot_seq_len:].real,symbs_out[0,pilot_seq_len:].imag,'.')
plt.title("Equalized received symbols, including phase pilots")
plt.figure()
plt.plot(symbs_out[0,0:pilot_seq_len-5].real,symbs_out[0,0:pilot_seq_len-5].imag,'.')
plt.title("Equalized Pilots")
plt.figure()
plt.plot(np.abs(err[0]))
plt.tite("CMA Error")