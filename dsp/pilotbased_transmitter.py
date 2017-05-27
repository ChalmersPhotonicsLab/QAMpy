#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:31:05 2017

Pilot-Based DSP Transmitter for MQAM with M>4

@author: mazurm
"""
import numpy as np
from dsp import signals, equalisation, modulation, utils, phaserecovery, dsp_cython, signal_quality
from scipy.io import loadmat, savemat
from dsp.prbs import make_prbs_extXOR

# Oversampling
os = 2
    
# Mapper settings
M = 128

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

data_modulator = modulation.QAMModulator(M)
pilot_modualtor = modulation.QAMModulator(4)

N_data_frames = (frame_length - pilot_seq_len) / pilot_ins_ratio

if (N_data_frames%1) != 0:
    print("Pilot insertion ratio not propper selected")

N_pilot_symbs = pilot_seq_len + N_data_frames

N_data_symbs = N_data_frames * (pilot_ins_ratio - 1)

N_pilot_bits = (N_pilot_symbs) * pilot_modualtor.bits
N_data_bits = N_data_symbs * data_modulator.bits

# Generate bits for modulation
if PRBS == True:
    pilot_bits = make_prbs_extXOR(PRBSorder, N_pilot_bits, PRBSseed)
    data_bits = make_prbs_extXOR(PRBSorder, N_data_bits, PRBSseed)
else:
    pilot_bits = np.random.randint(0, high=2, size=N_pilot_bits).astype(np.bool)    
    data_bits = np.random.randint(0, high=2, size=N_data_bits).astype(np.bool)

pilot_symbs = pilot_modualtor.modulate(pilot_bits)
data_symbs = data_modulator.modulate(data_bits)
 

# Set sequence together
symbol_seq = np.zeros(frame_length, dtype=np.complex)

# Insert pilot sequence
symbol_seq[0:pilot_seq_len] = pilot_symbs[0:pilot_seq_len]
symbol_seq\
[(pilot_seq_len)::pilot_ins_ratio] \
= pilot_symbs[pilot_seq_len:]

# Insert data symbols
for j in np.arange(N_data_frames):
    symbol_seq[pilot_seq_len + j*pilot_ins_ratio + 1:\
                    pilot_seq_len + (j+1)*pilot_ins_ratio ] = \
                    data_symbs[j*(pilot_ins_ratio-1):(j+1)*(pilot_ins_ratio-1)]
