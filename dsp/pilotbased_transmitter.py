#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:31:05 2017

Pilot-Based DSP Transmitter for MQAM with M>4

@author: mazurm
"""
import numpy as np
from dsp import signals, equalisation, modulation, utils, phaserecovery, dsp_cython, signal_quality
from dsp.prbs import make_prbs_extXOR
import matplotlib.pylab as plt


def gen_dataframe_with_phasepilots(M,npols, frame_length = 2**18, pilot_seq_len = 256, pilot_ins_ratio = 32, PRBS = False, PRBSorder=15, PRBSseed=None):

    data_modulator = modulation.QAMModulator(M)
    pilot_modualtor = modulation.QAMModulator(4)
    
    N_data_frames = (frame_length - pilot_seq_len) / pilot_ins_ratio
    
    if (N_data_frames%1) != 0:
        raise ValueError("Pilot insertion ratio not propper selected")
    
    N_pilot_symbs = pilot_seq_len + N_data_frames    
    N_data_symbs = N_data_frames * (pilot_ins_ratio - 1)    
    N_pilot_bits = (N_pilot_symbs) * pilot_modualtor.bits
    N_data_bits = N_data_symbs * data_modulator.bits
       
    # Set sequence together
    symbol_seq = np.zeros([npols,frame_length], dtype = complex)
    data_symbs = np.zeros([npols,N_data_symbs], dtype = complex)
    pilot_symbs = np.zeros([npols,N_pilot_symbs], dtype = complex)
    
    for l in range(npols):
    
        # Generate bits for modulation
        if PRBS == True:
            pilot_bits = make_prbs_extXOR(PRBSorder, N_pilot_bits, PRBSseed)
            data_bits = make_prbs_extXOR(PRBSorder, N_data_bits, PRBSseed)
        else:
            pilot_bits = np.random.randint(0, high=2, size=N_pilot_bits).astype(np.bool)    
            data_bits = np.random.randint(0, high=2, size=N_data_bits).astype(np.bool)
        
        pilot_symbs[l,:] = pilot_modualtor.modulate(pilot_bits)
        data_symbs[l,:] = data_modulator.modulate(data_bits)
        
        # Insert pilot sequence
        symbol_seq[l,0:pilot_seq_len] = pilot_symbs[l,0:pilot_seq_len]
        symbol_seq\
        [l,(pilot_seq_len)::pilot_ins_ratio] \
        = pilot_symbs[l,pilot_seq_len:]
        
        # Insert data symbols
        for j in np.arange(N_data_frames):
            symbol_seq[l,pilot_seq_len + j*pilot_ins_ratio + 1:\
                            pilot_seq_len + (j+1)*pilot_ins_ratio ] = \
                            data_symbs[l,j*(pilot_ins_ratio-1):(j+1)*(pilot_ins_ratio-1)]
    
    return symbol_seq, data_symbs, pilot_symbs


def gen_dataframe_without_phasepilots(M,npols, frame_length = 2**18, pilot_seq_len = 256, PRBS = False, PRBSorder=15, PRBSseed=None):

    data_modulator = modulation.QAMModulator(M)
    pilot_modualtor = modulation.QAMModulator(4)
    
    
    if (pilot_seq_len >= frame_length):
        raise ValueError("Pilot insertion ratio not propper selected")
    
    
    N_data_symbs = frame_length - pilot_seq_len
    
    N_pilot_bits = (pilot_seq_len) * pilot_modualtor.bits
    N_data_bits = N_data_symbs * data_modulator.bits
    
    
    # Set sequence together
    symbol_seq = np.zeros([npols,frame_length], dtype = complex)
    data_symbs = np.zeros([npols,N_data_symbs], dtype = complex)
    pilot_symbs = np.zeros([npols,pilot_seq_len], dtype = complex)
    
    for l in range(npols):
    
        # Generate bits for modulation
        if PRBS == True:
            pilot_bits = make_prbs_extXOR(PRBSorder, N_pilot_bits, PRBSseed)
            data_bits = make_prbs_extXOR(PRBSorder, N_data_bits, PRBSseed)
        else:
            pilot_bits = np.random.randint(0, high=2, size=N_pilot_bits).astype(np.bool)    
            data_bits = np.random.randint(0, high=2, size=N_data_bits).astype(np.bool)
        
        pilot_symbs[l,:] = pilot_modualtor.modulate(pilot_bits)
        data_symbs[l,:] = data_modulator.modulate(data_bits)
        
        # Insert pilot sequence
        symbol_seq[l,0:pilot_seq_len] = pilot_symbs[l,0:pilot_seq_len]

        # Insert data symbols
        symbol_seq[l,pilot_seq_len:] = data_symbs[l,:]
    
    return symbol_seq, data_symbs, pilot_symbs

def gen_dataframe_with_phasepilots_hybridmodulation(M=(128,256),mod_ratio = (1,1),npols=2, frame_length = 2**18, pilot_seq_len = 256, pilot_ins_ratio = 32):


    if len(M) != len(mod_ratio):
        raise ValueError("Number of moduation formats and insertion ratios does not match")

    
    N_data_frames = (frame_length - pilot_seq_len) / pilot_ins_ratio
    if (N_data_frames%1) != 0:
        raise ValueError("Pilot insertion ratio not propper selected")

    # Modulators to generate hybrid QAM
    pilot_modualtor = modulation.QAMModulator(4)
    data_modulators = []
    for mod in M:
        data_modulators.append(modulation.QAMModulator(mod))
        
    # Arrange to have same average power of both
    N_data_symbs = int(N_data_frames * (pilot_ins_ratio - 1))
    norm_factors = np.zeros(len(M))
    for i in range(len(M)):
        norm_factors[i] = np.min(np.abs(data_modulators[i].symbols[0]-data_modulators[i].symbols[1:]))/2

    # Pilot symbols    
    N_pilot_symbs = pilot_seq_len + N_data_frames    
    N_pilot_bits = (N_pilot_symbs) * pilot_modualtor.bits

    # Number of symbols per modulation format, correction added later
    sub_blocks = N_data_symbs // sum(mod_ratio)
    rem_symbs = N_data_symbs % sum(mod_ratio)

    # Set sequence together
    symbol_seq = np.zeros([npols,frame_length], dtype = complex)
    data_symbs = np.zeros([npols,N_data_symbs], dtype = complex)
    pilot_symbs = np.zeros([npols,N_pilot_symbs], dtype = complex)

    for l in range(npols):
        # Generate modulated symbols
        mod_symbs = []
        for i in range(len(M)):
            if rem_symbs- sum(mod_ratio[:i+1]) > 0:
                if sum(mod_ratio[:i+1]) < rem_symbs:
                    add_symbs = sum(mod_ratio[:i+1])
                else:
                    add_symbs = rem_symbs - sum(mod_ratio[:i])
                    
                tmp_bits = np.random.randint(0,high=2, size = (sub_blocks*mod_ratio[i]+add_symbs)*M[i]).astype(np.bool)
            else:
                tmp_bits = np.random.randint(0,high=2, size = sub_blocks*mod_ratio[i]*M[i]).astype(np.bool)
            
            mod_symbs.append(data_modulators[i].modulate(tmp_bits)/norm_factors[i]) 
        
        place_ind = np.cumsum(mod_ratio)
        for i in range(N_data_symbs):
            place = np.argmax(i%place_ind != 0)
            data_symbs[l,i] = mod_symbs[place][i//sum(mod_ratio)]
        
        # Normalize output symbols to have a unit energy of 1
        data_symbs[l,:] = data_symbs[l,:]/np.sqrt(np.mean(np.abs(data_symbs[l,:])**2))        
        
        # Modulate the pilot symbols
        pilot_bits = np.random.randint(0, high=2, size=N_pilot_bits).astype(np.bool)
        pilot_symbs[l,:] = pilot_modualtor.modulate(pilot_bits)
        
        # Insert pilot sequence
        symbol_seq[l,0:pilot_seq_len] = pilot_symbs[l,0:pilot_seq_len]
        symbol_seq\
        [l,(pilot_seq_len)::pilot_ins_ratio] \
        = pilot_symbs[l,pilot_seq_len:]
        
        # Insert data symbols
        for j in np.arange(N_data_frames):
            symbol_seq[l,pilot_seq_len + j*pilot_ins_ratio + 1:\
                            pilot_seq_len + (j+1)*pilot_ins_ratio ] = \
                            data_symbs[l,j*(pilot_ins_ratio-1):(j+1)*(pilot_ins_ratio-1)]
    
    return symbol_seq, data_symbs, pilot_symbs
