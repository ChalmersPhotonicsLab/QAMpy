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

def pilot_based_foe(rec_symbs,pilot_symbs, dual_pol = 0 ):
    """
    Frequency offset estimation for pilot-based DSP. Uses a transmitted pilot
    sequence to find the frequency offset from the corresponding aligned symbols.
    
    Gives higher accuracy than blind power of 4 based FFT for noisy signals. 
    Calculates the phase variations between the batches and does a linear fit
    to find the corresponding frequency offset. 
    
    Input:
        rec_symbs:  Complex symbols after initial Rx DSP
        pilot_symbs: Complex pilot symbols transmitted
        
    
    Output:
        foe:    Estimated frequency offset. Outputed in terms of complex phase
        cond:   Condition number of linear fit. Gives accuracy of estimation
    
    """

    phaseEvolution = np.unwrap(np.angle(pilot_symbs.conj()*rec_symbs))
    
    # fit a first order polynomial to the unwrapped phase evolution
    freqFit = np.polyfit(np.arange(0,len(phaseEvolution)),phaseEvolution,1)
    
    freqFound = freqFit[0]/(2*np.pi)
    condNum = freqFit[1]                 
                       
    return freqFound, condNum

def pilot_based_cpe(rec_symbs, pilot_symbs, pilot_ins_ratio, num_average = 3, use_pilot_ratio = 1):
    """
    Carrier phase recovery using periodically inserted symbols.
    
    Performs a linear interpolation with averaging over n symbols to estimate
    the phase drift from laser phase noise to compensate for this.
    
    Input: 
        rec_symbs: Received symbols in block (first of each block is the pilot)
        pilot_symbs: Corresponding pilot symbols. 
            Index N is the first symbol in transmitted block N.
        pilot_ins_ratio: Length of each block. Ex. 16 -> 1 pilot symbol followed
            by 15 data symbols
        num_average: Number of pilot symbols to average over to avoid noise. 
        use_pilot_ratio: Use ever n pilots. Can be used to sweep required rate.
        
    Output:
        data_symbs: Complex symbols after pilot-aided CPE. Pilot symbols removed
    """
    # Extract the pilot symbols
    rec_pilots = rec_symbs[::pilot_ins_ratio]
    
    # Remove every X pilot symbol if wanted
    rec_pilots = rec_pilots[::use_pilot_ratio]
    
    
    # If not a complete block, remove the remaning symbols
    if len(pilot_symbs)> np.floor(len(rec_symbs)/(pilot_ins_ratio*use_pilot_ratio)):
        pilot_symbs = pilot_symbs[:len(rec_symbs)]
        
    # Should be an odd number to keey symmetry in averaging
    if (num_average % 2):
        num_average += 1
    
    # Calculate phase respons
    res_phase = pilot_symbs.conjugate()*rec_pilots
    pilot_phase = np.unwrap(np.angle(res_phase))
    
    # Fix! Need moving average in numpy
    pilot_phase_average = moving_average(pilot_phase,num_average)
    pilot_phase = [pilot_phase[:(num_average-1) / 2], pilot_phase_average]
    
    
    # Pilot positions in the received data set
    pilot_pos = np.arange(0,len(pilot_phase),pilot_ins_ratio*use_pilot_ratio)
    
    # Lineary interpolate the phase evolution
    phase_evol = np.interp(np.arange(0,len(pilot_phase)*pilot_ins_ratio),\
                           pilot_pos,pilot_phase)
    
    # Compensate phase
    comp_symbs = rec_symbs*np.exp(-1j*phase_evol)
    
    # Allocate output memory
    out_size = np.shape(rec_symbs)
    data_symbs = np.zeros(out_size[0],out_size[1]-len(pilot_symbs))
    
    # Allocate output by removing pilots
    block_len = (pilot_ins_ratio*use_pilot_ratio) - 1
    for i in range(len(pilot_symbs)):
        data_symbs[i*block_len:(i+1)*block_len] = \
                   comp_symbs[i*(pilot_ins_ratio*use_pilot_ratio):\
                              (i+1)*(pilot_ins_ratio*use_pilot_ratio)]
    
    # If additional pilots are in between, throw them out
    pilot_pos = np.arange(0,len(pilot_symbs),pilot_ins_ratio)
    for i in range(0,len(pilot_pos)):
        if (pilot_pos[i]%(pilot_ins_ratio*use_pilot_ratio)) == 0:
            print(pilot_pos[i])
            pilot_pos[i] = 0
            
    pilot_pos = pilot_pos[pilot_pos != 0]
    
    data_symbs = np.delete(data_symbs,pilot_pos)
    
    
    return data_symbs
    
    
def moving_average(sig, n=3):
    """
    Moving average of signal
    
    Input:
        sig: Signal for moving average
        n: number of averaging samples
        
    Output:
        ret: Returned average signal of length len(sig)-n+1
    """
    
    ret = np.cumsum(sig,dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n-1:]/n


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

def frame_sync(rx_signal, os, pilot_seq_length = 256, mu = 1e-3, M_pilot = 4, Ntaps = 25, Niter = 10, adap_step = True):
    """
    Locate the pilot sequence starting the frame and syncronize it
    
    """
    
    symb_step_size = int(np.floor(pilot_seq_len / 2 * os))
    num_steps = int(np.ceil(frame_length / symb_step_size))

    sub_var = np.zeros([1,num_steps])
    
    for i in np.arange(0,num_steps):
        wx, err = equalisation.equalise_signal(rx_signal[(i)*symb_step_size:(i+1)*symb_step_size], os, mu, M_pilot,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step) 
        sub_var[0,i] = np.var(err[0,-symb_step_size/os+N_taps:])
        
    minPart = np.argmin(sub_var)
    
    

    # Extract a longer sequence
    seq = rx_signal[(minPart-2)*symb_step_size:(minPart+3)*symb_step_size]
        
    # Now test for all polarizations, first pre-convergence from before
    wx, err = equalisation.equalise_signal(rx_signal[(minPart)*symb_step_size:(minPart+1)*symb_step_size], os, mu, M_pilot,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step)    
    wx, err = equalisation.equalise_signal(seq, os, mu/10, M_pilot,wxy=wx,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = True) 
    
    
    
    symbs_out= equalisation.apply_filter(seq,os,wx)

    
    #Find the variations
    xcov = np.correlate(np.angle(symbs_out[0,:]),np.angle(ref_symbs))
    symb_delay = np.argmax(xcov)
    
    # Extract the received pilots
    rec_pilots = symbs_out[0,symb_delay:symb_delay+pilot_seq_len]
    
    # New starting sample
    shift_factor = (minPart-2)*symb_step_size +os*symb_delay
    
    
    # Verification and plotting
    test = equalisation.apply_filter(rx_signal[shift_factor:shift_factor+pilot_seq_len*os+N_taps-1],os,wx)
    plt.plot(test.real,test.imag,'.')
    
    plt.plot(rec_pilots.real,rec_pilots.imag,'.')
    
    
    a = rec_pilots - ref_symbs
    plt.plot(a.imag)
    
    
    
    

tx_sig_shift = np.roll(tx_sig,int(1e4))

numTests = frame_length/pilot_seq_len

batchVar = np.zeros([numTests,1])


mu = 1e-3
M_pilot = 4
N_taps = 25
adap_step = True
Niter = 10

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