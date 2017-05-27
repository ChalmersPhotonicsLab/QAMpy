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
pilot_sequence_length = 256

# Repetative pilot symbols for phase tracking and equalizer update
pilot_insertion_ratio = 32 

# Settings if PRBS is seleted for generation
PRBS = False
PRBSorder=15,
PRBSseed=None

data_modulator = modulation.QAMModulator(M)
pilot_modualtor = modulation.QAMModulator(4)


N_pilot_symbs = pilot_sequence_length + (frame_length - pilot_sequence_length) / pilot_insertion_ratio 

if (N_pilot_symbs%1) != 0:
    print("Pilot insertion ratio not propper selected")

N_data_symbs = frame_length - N_pilot_symbs

N_pilot_bits = (N_pilot_symbs) * pilot_modualtor.bits
N_data_bits = N_data_symbs * data_modulator.bits

# Generate bits for modulation
if PRBS == True:
    pilot_bits = make_prbs_extXOR(PRBSorder, N_pilot_bits, PRBSseed)
    data_bits = make_prbs_extXOR(PRBSorder, N_data_bits, PRBSseed)
else:
    pilot_bits = np.random.randint(0, high=2, size=N_pilot_bits).astype(np.bool)    
    data_bits = np.random.randint(0, high=2, size=N_data_bits).astype(np.bool)

pilot_symbols = pilot_modualtor.modulate(pilot_bits)
data_symbols = data_modulator.modulate(data_bits)
 

# Set sequence together
symbol_sequence = np.zeros(frame_length, dtype=np.complex)

# Insert pilot sequence
symbol_sequence[0:pilot_sequence_length] = pilot_symbols[0:pilot_sequence_length]
symbol_sequence\
[(pilot_sequence_length)::pilot_insertion_ratio] \
= pilot_symbols[pilot_sequence_length:]

# Insert data symbols
for j in np.arange(N_pilot_symbs - pilot_sequence_length):
    symbol_sequence[pilot_sequence_length + j*pilot_insertion_ratio + 1:\
                    pilot_sequence_length + (j+1)*pilot_insertion_ratio ] = \
                    data_symbols[j*(pilot_insertion_ratio-1):(j+1)*(pilot_insertion_ratio-1)]
