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

def pilot_based_foe(rec_symbs,pilot_symbs):
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
        foe:    Estimated FO in terms of complex phase. Average over all modes
        foePerMode: FO estimate for each mode
        condNum:   Condition number of linear fit. Gives accuracy of estimation
    
    """

    rec_symbs = np.atleast_2d(rec_symbs)
    pilot_symbs = np.atleast_2d(pilot_symbs)
    npols = rec_symbs.shape[0]
    
    condNum = np.zeros([npols,1])
    foePerMode = np.zeros([npols,1])
    
    # Search over all polarization
    for l in range(npols):    
        phaseEvolution = np.unwrap(np.angle(pilot_symbs[l,:].conj()*rec_symbs[l,:]))
        
        # fit a first order polynomial to the unwrapped phase evolution
        freqFit = np.polyfit(np.arange(0,len(phaseEvolution)),phaseEvolution,1)
    
        foePerMode[l,0] = freqFit[0]/(2*np.pi)
        condNum[l,0] = freqFit[1]                 
    
    # Average over all modes used
    foe = np.mean(foePerMode)
    
    return foe, foePerMode, condNum

def pilot_based_cpe(rec_symbs, pilot_symbs, pilot_ins_ratio, num_average = 3, use_pilot_ratio = 1, max_num_blocks = None):
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
        max_num_blocks: Maximum number of blocks to process
        
    Output:
        data_symbs: Complex symbols after pilot-aided CPE. Pilot symbols removed
    """
    
    rec_symbs = np.atleast_2d(rec_symbs)
    pilot_symbs = np.atleast_2d(pilot_symbs)
    npols = rec_symbs.shape[0]
    
    
    # Extract the pilot symbols
    numBlocks = np.floor(np.shape(rec_symbs)[1]/pilot_ins_ratio)
    rec_pilots = rec_symbs[:,::pilot_ins_ratio] 
    rec_pilots = rec_pilots[:,:numBlocks]
    rec_symbs = rec_symbs[:,:pilot_ins_ratio*numBlocks]
    

    # If selected, only process a limited number of blocks. 
    if (max_num_blocks is not None) and numBlocks > max_num_blocks:
        numBlocks = max_num_blocks    

    
    numRefPilots = np.shape(pilot_symbs)[1]   

    if numBlocks > numRefPilots:
        print("HEJ")
        print("HEJ")
        numBlocks = numRefPilots
        rec_symbs = rec_symbs[:,numBlocks*pilot_ins_ratio]
        rec_pilots = rec_pilots[:,:numBlocks]
    elif numRefPilots > numBlocks:
        print("HEJ")
        print("HEJ")
        pilot_symbs = pilot_symbs[:,:numBlocks]
    
    # Remove every X pilot symbol if selected
    rec_pilots = rec_pilots[:,::use_pilot_ratio]
    pilot_symbs = pilot_symbs[:,::use_pilot_ratio]    
        
    
    # Should be an odd number to keey symmetry in averaging
    if not(num_average % 2):
        num_average += 1
    
    # Allocate output memory
    data_symbs = np.zeros([npols,len(rec_symbs[0,:]) -len(pilot_symbs[0,:])], dtype = complex)
    data_symbs = np.zeros([npols,len(rec_symbs[0,:])], dtype = complex)
    
    for l in range(npols):
    
        # Calculate phase respons
        res_phase = pilot_symbs[l,:].conjugate()*rec_pilots[l,:]
        pilot_phase = np.unwrap(np.angle(res_phase))
        
        plt.plot(pilot_phase)
        
        # Fix! Need moving average in numpy
#        pilot_phase_average = np.transpose(moving_average(pilot_phase,num_average))
#        pilot_phase = [pilot_phase[:(num_average-1) / 2], pilot_phase_average]
           
        # Pilot positions in the received data set
        pilot_pos = np.arange(0,len(pilot_phase)*pilot_ins_ratio*use_pilot_ratio,pilot_ins_ratio*use_pilot_ratio)
        


        # Lineary interpolate the phase evolution
        phase_evol = np.interp(np.arange(0,len(pilot_phase)*pilot_ins_ratio),\
                               pilot_pos,pilot_phase)

        
        # Compensate phase
        comp_symbs = rec_symbs[l,:]*np.exp(-1j*phase_evol)

        # Allocate output by removing pilots
        block_len = (pilot_ins_ratio*use_pilot_ratio)
        for i in range(np.shape(pilot_symbs)[1]):
            print(i)
            print(np.shape(comp_symbs))
            data_symbs[l,i*(block_len):(i+1)*(block_len)] = \
                       comp_symbs[i*block_len:(i+1)*block_len]
   
    
    # If additional pilots are in between, throw them out
    pilot_pos = np.arange(0,len(pilot_symbs[0,:]),pilot_ins_ratio)
    for i in range(0,len(pilot_pos)):
        if (pilot_pos[i]%(pilot_ins_ratio*use_pilot_ratio)) == 0:
            print(pilot_pos[i])
            pilot_pos[i] = 0
            
    pilot_pos = pilot_pos[pilot_pos != 0]    
    data_symbs = np.delete(data_symbs,pilot_pos, axis = 1)   
    
    return data_symbs, pilot_pos, comp_symbs
    
    
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



"""
 
Locate pilot sequence

"""

def frame_sync(rx_signal, ref_symbs, os, mu = 1e-3, M_pilot = 4, ntaps = 25, Niter = 10, adap_step = True):
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
        wx: Taps for equalization of the whole signal
    
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
        wx, err = equalisation.equalise_signal(rx_signal[:,(i)*symb_step_size:(i+1)*symb_step_size], os, mu, M_pilot,Ntaps = ntaps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step) 
        sub_var[:,i] = np.var(err[:,-symb_step_size/os+ntaps:],axis = 1)
    
    
    # Now search for every mode independent
    eq_pilots = np.zeros([npols,pilot_seq_len],dtype = complex)
    shift_factor = np.zeros([npols,1],dtype = int)
    for l in range(npols):
        
        # Lowest variance of the CMA error
        minPart = np.argmin(sub_var[l,:])
        
        # Corresponding sequence
        shortSeq = rx_signal[:,(minPart)*symb_step_size:(minPart+1)*symb_step_size]
        
        # Extract a longer sequence to ensure that the complete pilot sequence is found
        longSeq = rx_signal[:,(minPart-2)*symb_step_size:(minPart+3)*symb_step_size]

        # Use the first estimate to get rid of any large FO and simplify alignment
        wx1, err = equalisation.equalise_signal(shortSeq, os, mu, M_pilot,Ntaps = ntaps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step)    
        seq_foe = equalisation.apply_filter(longSeq,os,wx1)
        foe_corse = phaserecovery.find_freq_offset(seq_foe)        
         
        # Apply filter taps to the long sequence
        symbs_out= equalisation.apply_filter(longSeq,os,wx1)     
        symbs_out[l,:] = phaserecovery.comp_freq_offset(symbs_out[l,:], foe_corse[l,:])
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # For DEBUG!!!
        test_out = equalisation.apply_filter(shortSeq,os,wx1)
        test_out[l,:] = phaserecovery.comp_freq_offset(test_out[l,:], foe_corse[l,:])
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
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
        pilot_seq = rx_signal[:,shift_factor:shift_factor+pilot_seq_len*os+ntaps-1]
        wx1, err = equalisation.equalise_signal(pilot_seq, os, mu, M_pilot,Ntaps = ntaps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step) 
        wx, err = equalisation.equalise_signal(pilot_seq, os, mu/10, M_pilot,wxy=wx1,Ntaps = ntaps, Niter = Niter, method = "cma",adaptive_stepsize = adap_step) 
        symbs_out= equalisation.apply_filter(pilot_seq,os,wx)
#        symbs_out[l,:] = phaserecovery.comp_freq_offset(symbs_out[l,:], foe_corse, dual_pol = False) 
        eq_pilots[l,:] = symbs_out[l,:]
    

    return eq_pilots, shift_factor, wx, foe_corse, test_out

def find_const_phase_offset(rec_pilots, ref_symbs):
    """
    Finds and corrects a constant phase offset between the decoded pilot 
    symbols and the transmitted ones
    
    Input:
        rec_pilots: Complex received pilots (after FOE and alignment)
        ref_symbs: Corresponding transmitted pilot symbols (aligned!)
        
    Output:
        phase_corr_pilots: Phase corrected pilot symbols
        phase_corr: Corresponding phase offset per mode
    
    """
    
    rec_pilots = np.atleast_2d(rec_pilots)
    ref_symbs = np.atleast_2d(ref_symbs)
    npols = rec_pilots.shape[0]

    phase_corr = np.zeros([npols,1],dtype = float)
    
    for l in range(npols):    
        phase_corr[l] = np.mean(np.angle(ref_symbs[l,:].conj()*rec_pilots[l,:]))

    return  phase_corr
    

def correct_const_phase_offset(symbs, phase_offsets):
    symbs = np.atleast_2d(symbs)
    phase_offsets = np.atleast_2d(phase_offsets)
    npols = symbs.shape[0]

    for l in range(npols):
        symbs[l,:] = symbs[l,:] * np.exp(-1j*phase_offsets[l,0])

    return symbs


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


# Dynamic pilot_based equalization
#def pilotbased_equalization(rx_signal,os,wx, start_point ):

#Code for testing using the transmitter
    
#rec_signal = np.roll(tx_sig2[0,:], 7400)
rec_signal = tx_sig
ref_symbs = pilot_symbs[0,0:256]

# Frame sync
eq_pilots, shift_factor , wx, corse_foe,  test_out = frame_sync(rec_signal, ref_symbs, os)

tmp_signal = rec_signal[:,shift_factor:shift_factor+45]

# Foe estimate
foe, foePerMode, condNum = pilot_based_foe(eq_pilots,ref_symbs)

# Remove FOE from pilots
comp_test = phaserecovery.comp_freq_offset(eq_pilots[0,:],foe)

# Correct static phase difference between the Tx and Rx pilots
phase_offset = find_const_phase_offset(comp_test,ref_symbs)

corr_pilots = correct_const_phase_offset(comp_test,phase_offset)

#  Verification and plotting    
plt.plot(eq_pilots[0,:].real,eq_pilots[0,:].imag,'.')  

plt.plot(comp_test[0,:].real,comp_test[0,:].imag,'.')  

plt.plot(corr_pilots[0,:].real,corr_pilots[0,:].imag,'.')  

#Extract the real signal
test_sig = equalisation.apply_filter(rec_signal[:,shift_factor:],os,wx)
comp_test_sig = phaserecovery.comp_freq_offset(test_sig[0,:],foe)
comp_test_sig = correct_const_phase_offset(comp_test_sig,phase_offset)
plt.plot(comp_test_sig[0,:1000].real,comp_test_sig[0,:1000].imag,'.')

phase_comp_symbs, pil_pos, comp_symbs = pilot_based_cpe(comp_test_sig[0,256:],pilot_symbs[0,256:], pilot_ins_ratio, num_average = 1)

plt.hexbin(phase_comp_symbs[0,:].real, phase_comp_symbs[0,:].imag)





#wx1, err = equalisation.equalise_signal(pilot_seq, os, mu, 32,Ntaps = ntaps, Niter = Niter, method = "mcma",adaptive_stepsize = adap_step) 


#a = rec_pilots - ref_symbs
#plt.plot(a.imag)
#    
#mu = 1e-3
#M_pilot = 4
#N_taps = 25
#adap_step = True
#Niter = 10
#
#
#
#plt.figure()
#plt.plot(symbs_out[0,pilot_seq_len:].real,symbs_out[0,pilot_seq_len:].imag,'.')
#plt.title("Equalized received symbols, including phase pilots")
#plt.figure()
#plt.plot(symbs_out[0,0:pilot_seq_len-5].real,symbs_out[0,0:pilot_seq_len-5].imag,'.')
#plt.title("Equalized Pilots")
#plt.figure()
#plt.plot(np.abs(err[0]))
#plt.tite("CMA Error")