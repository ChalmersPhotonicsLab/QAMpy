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
import matplotlib.pylab as plt




def gen_dataframe_withpilots(M,npols, frame_length = 2**18, pilot_seq_len = 256, pilot_ins_ratio = 32, PRBS = False, PRBSorder=15, PRBSseed=None):

    data_modulator = modulation.QAMModulator(M)
    pilot_modualtor = modulation.QAMModulator(4)
    
    N_data_frames = (frame_length - pilot_seq_len) / pilot_ins_ratio
    
    if (N_data_frames%1) != 0:
        print("Pilot insertion ratio not propper selected")
    
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


def sim_tx(frame, os, symb_rate = 20e9, beta = 0.1, snr = None, linewidth = None, freqoff = None):
    """
    Function to simulate transmission distortions to pilot frame
    
    """
    
    npols = frame.shape[0]
    sig = np.zeros([npols,len(frame[0,:])*os],dtype = complex)
    
    for l in range(npols):
    
        # Upsample and pulse shaping
        if os > 1:
            sig[l,:] = utils.rrcos_resample_zeroins(frame[l,:], 1, os, beta = beta, renormalise = True)
        
        # Add AWGN
        if snr is not None:
            sig[l,:] = utils.add_awgn(sig[l,:],10**(-snr/20)*np.sqrt(os))
        
        # Add Phase Noise
        if linewidth is not None:       
            sig[l,:] = sig[l,:]*np.exp(1j*utils.phase_noise(sig[l,:].shape, linewidth, symb_rate * os ))
        
        # Add FOE
        if freqoff is not None:
            sig[l,:] *= np.exp(2.j * np.pi * np.arange(len(sig[l,:])) * freqoff / (symb_rate * os))
        
        # Verfy normalization
        sig = utils.normalise_and_center(sig)
    

    
    return sig

""" 
Testing the transmitter
"""
# Tx Config
M = 64
os = 2
symb_rate = 20e9
snr = 20 #dB
linewidth = None # Linewidth symbol-rate product
fo = 10 # Frequency offset MHz out

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


frame_symbs, data_symbs, pilot_symbs = gen_dataframe_withpilots(32,1)

frame_symbs = np.roll(frame_symbs, 6523)

tx_sig = sim_tx(frame_symbs, os,freqoff = 1.7e9)


"""
Plotting section to check transmitter
"""

# Plot Result
plt.figure()
plt.semilogy(np.fft.fftshift(np.fft.fftfreq(len(tx_sig[0,:]),1/os)), np.fft.fftshift(np.abs(np.fft.fft(tx_sig[0,:]))**2))

#plt.figure()
#plt.plot([symbol_seq[:100].real,symbol_seq[:100].imag],'.')



