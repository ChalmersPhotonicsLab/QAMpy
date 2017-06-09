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
    if not(num_average % 2):
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

def frame_sync(rx_signal, ref_symbs, os, mu = 1e-3, M_pilot = 4, Ntaps = 25, Niter = 10, adap_step = True):
    """
    Locate and extract the pilot starting frame.
    
    Uses a CMA-based search scheme to located the initiial pilot sequence in
    the long data frame. 
    
    Input:
        rx_signal: Received Rx signal
        ref_symbs: Pilot sequence
        os: Oversampling
        mu: CMA step size
        M_pilot: Order for pilot symbols. Should normally be QPSK
        Ntaps: Number of T/2-spaced taps for equalization
        Niter: Number of iterations for the equalizer
        adap_step: Use adaptive step size (bool)
        
    Output:
        eq_pilots: Found pilot sequence after equalization
        shift_factor: New starting point for initial equalization
    
    """
    # Fix number of stuff
    rx_signal = np.atleast_2d(rx_signal)
    ref_symbs = np.atleast_2d(ref_symbs)
    npols = rx_signal.shape[0]
    
    # Find the length of the pilot frame
    pilot_seq_len = len(ref_symbs[0,:])
    
    symb_step_size = int(np.floor(pilot_seq_len / 2 * os))
    num_steps = int(np.ceil(frame_length / symb_step_size))

    # Search based on equalizer error. Avoid certain part in the beginning and
    # end to ensure that sufficient symbols can be used for the search
    sub_var = np.ones([npols,num_steps])*1e2        
    for i in np.arange(2,num_steps-3):
        wx, err = equalisation.equalise_signal(rx_signal[:,(i)*symb_step_size:(i+1)*symb_step_size], os, mu, M_pilot,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step) 
        sub_var[:,i] = np.var(err[:,-symb_step_size/os+N_taps:],axis = 1)
    
    
    # Now search for every mode independent
    eq_pilots = np.zeros([npols,pilot_seq_len],dtype = complex)
    shift_factor = np.zeros([npols,1],dtype = int)
    for l in range(npols):
        
        # Lowest variance of the CMA error
        minPart = np.argmin(sub_var[l,:])
        
        # Extract a longer sequence to ensure that the complete pilot sequence is found
        seq = rx_signal[:,(minPart-2)*symb_step_size:(minPart+3)*symb_step_size]
            
        # Now test for all polarizations, first pre-convergence from before
        wx1, err = equalisation.equalise_signal(rx_signal[:,(minPart)*symb_step_size:(minPart+1)*symb_step_size], os, mu, M_pilot,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step)    
        wx2, err = equalisation.equalise_signal(seq, os, mu/10, M_pilot,wxy=wx1,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = True) 
        
        # Apply filter taps to the long sequence
        symbs_out= equalisation.apply_filter(seq,os,wx2)
        
        # Check for pi/2 ambiguties
        max_phase_rot = np.zeros([1,4])
        found_delay = np.zeros([1,4])
        for k in range(4):
            # Find correlation for all 4 possible pi/2 rotations
            xcov = np.correlate(np.angle(symbs_out[l,:]*np.exp(1j*k)),np.angle(ref_symbs[l,:]))
            max_phase_rot[0,k] = np.max(xcov)
            found_delay[0,k] = np.argmax(xcov)
    
        # Select the best one    
        symb_delay = int(found_delay[0,np.argmax(max_phase_rot[0,:])]) 
        
        # New starting sample
        shift_factor[l,:] = int((minPart-2)*symb_step_size + os*symb_delay)
        
        # Tap update and extract the propper pilot sequuence
        pilot_seq = rx_signal[:,shift_factor:shift_factor+pilot_seq_len*os+N_taps-1]
        wx, err = equalisation.equalise_signal(pilot_seq, os, mu/10, M_pilot,wxy=wx1,Ntaps = N_taps, Niter = Niter, method = "cma",adaptive_stepsize = True) 
        symbs_out= equalisation.apply_filter(pilot_seq,os,wx)
        eq_pilots[l,:] = symbs_out[l,:]    
    
    return eq_pilots, shift_factor


#  Verification and plotting    
plt.plot(eq_pilots[l,:].real,eq_pilots[l,:].imag,'.')  

a = rec_pilots - ref_symbs
plt.plot(a.imag)
    
mu = 1e-3
M_pilot = 4
N_taps = 25
adap_step = True
Niter = 10



plt.figure()
plt.plot(symbs_out[0,pilot_seq_len:].real,symbs_out[0,pilot_seq_len:].imag,'.')
plt.title("Equalized received symbols, including phase pilots")
plt.figure()
plt.plot(symbs_out[0,0:pilot_seq_len-5].real,symbs_out[0,0:pilot_seq_len-5].imag,'.')
plt.title("Equalized Pilots")
plt.figure()
plt.plot(np.abs(err[0]))
plt.tite("CMA Error")