# -*- coding: utf-8 -*-
#  This file is part of QAMpy.
#
#  QAMpy is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Foobar is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with QAMpy.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2018 Jochen Schröder, Mikael Mazur


import numpy as np
from scipy.interpolate import interp1d
from qampy.core import equalisation
from qampy.core import phaserecovery, filter
from qampy.core import ber_functions


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

def pilot_based_cpe2(rec_symbs, pilot_symbs,  num_average = 1, use_pilot_ratio = 1):
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
        remove_phase_pilots: Remove phase pilots after CPE. Default: True

    Output:
        data_symbs: Complex symbols after pilot-aided CPE. Pilot symbols removed
        phase_trace: Resulting phase trace of the CPE
    """

    #rec_symbs = np.atleast_2d(rec_symbs)
    #pilot_symbs = np.atleast_2d(pilot_symbs)
    pilots_symbs = rec_symbs.extract_pilots()
    npols = rec_symbs.shape[0]

    # Extract the pilot symbols
    #numBlocks = np.floor(np.shape(rec_symbs)[1]/pilot_ins_ratio)
    # If selected, only process a limited number of blocks.
    #if (max_num_blocks is not None) and numBlocks > max_num_blocks:
    #    numBlocks = max_num_blocks

    # Make sure that a given number of pilots can be used
    if (numBlocks % use_pilot_ratio):
        numBlocks -= (numBlocks % use_pilot_ratio)

    # Adapt for number of blocks
    #rec_pilots = rec_symbs[:,::int(pilot_ins_ratio)]
    #rec_pilots = rec_pilots[:,:int(numBlocks)]
    #rec_symbs = rec_symbs[:,:int(pilot_ins_ratio*numBlocks)]

    # Check that the number of blocks are equal and is valid
    numRefPilots = np.shape(pilot_symbs)[1]
    if numBlocks > numRefPilots:
        numBlocks = numRefPilots
        rec_symbs = rec_symbs[:,:int(numBlocks*pilot_ins_ratio)]
        rec_pilots = rec_pilots[:,:int(numBlocks)]
    elif numRefPilots > numBlocks:
        pilot_symbs = pilot_symbs[:,:int(numBlocks)]

    # Remove every X pilot symbol if selected
    if use_pilot_ratio >= pilot_symbs.shape[1]:
        raise ValueError("Can not use every %d pilots since only %d pilot symbols are present"%(use_pilot_ratio,pilot_symbs.shape[1]))
    rec_pilots = rec_pilots[:,::int(use_pilot_ratio)]
    pilot_symbs = pilot_symbs[:,::int(use_pilot_ratio)]

    # Check for a out of bounch error
    if pilot_symbs.shape[1] <= num_average:
        raise ValueError("Inpropper pilot symbol configuration. Larger averaging block size than total number of pilot symbols")

    # Should be an odd number to keey symmetry in averaging
    if not(num_average % 2):
        num_average += 1

    # Allocate output memory and process modes
    data_symbs = np.zeros([npols,np.shape(rec_symbs)[1]], dtype = complex)
    phase_trace = np.zeros([npols,np.shape(rec_symbs)[1]])
    res_phase = pilot_symbs.conjugate()*rec_pilots
    pilot_phase_base = np.unwrap(np.angle(res_phase), axis=-1)
    pilot_phase_average = filter.moving_average(pilot_phase_base,num_average)
    pp1 = pilot_phase_base[:, :int((num_average-1)/2)]
    pp2 = pilot_phase_average[:]
    t = pilot_phase_base[:,:int((num_average-1)/2)]
    t2 = pp2[:, -1].reshape(2,1)
    pp3 = np.ones(t.shape)*t2
    pilot_phase = np.hstack([pp1, pp2, pp3])
    pilot_pos = np.arange(0,pilot_phase.shape[-1]*pilot_ins_ratio*use_pilot_ratio, pilot_ins_ratio*use_pilot_ratio)
    pilot_pos_new = np.arange(0,pilot_phase.shape[-1]*pilot_ins_ratio*use_pilot_ratio)

    for l in range(npols):
        phase_trace[l,:] = np.interp(pilot_pos_new, pilot_pos, pilot_phaseN[l])
    data_symbs = rec_symbs*np.exp(-1j*phase_trace)

    # Remove the phase pilots after compensation. This is an option since they can be used for SNR estimation e.g.
    if remove_phase_pilots:
        pilot_pos = np.arange(0,np.shape(data_symbs)[1],pilot_ins_ratio)
        data_symbs = np.delete(data_symbs,pilot_pos, axis = 1)

    return data_symbs, phase_trace

def pilot_based_cpe(rec_symbs, pilot_symbs, pilot_ins_ratio, num_average = 1, use_pilot_ratio = 1, max_num_blocks = None, remove_phase_pilots = True):
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
        remove_phase_pilots: Remove phase pilots after CPE. Default: True
        
    Output:
        data_symbs: Complex symbols after pilot-aided CPE. Pilot symbols removed
        phase_trace: Resulting phase trace of the CPE
    """
    
    rec_symbs = np.atleast_2d(rec_symbs)
    pilot_symbs = np.atleast_2d(pilot_symbs)
    npols = rec_symbs.shape[0]

    # Extract the pilot symbols
    numBlocks = np.floor(np.shape(rec_symbs)[1]/pilot_ins_ratio)
    # If selected, only process a limited number of blocks. 
    if (max_num_blocks is not None) and numBlocks > max_num_blocks:
        numBlocks = max_num_blocks   
    
    # Make sure that a given number of pilots can be used
    if (numBlocks % use_pilot_ratio):
        numBlocks -= (numBlocks % use_pilot_ratio)            

    # Adapt for number of blocks
    rec_pilots = rec_symbs[:,::int(pilot_ins_ratio)]
    rec_pilots = rec_pilots[:,:int(numBlocks)]
    rec_symbs = rec_symbs[:,:int(pilot_ins_ratio*numBlocks)]

    # Check that the number of blocks are equal and is valid
    numRefPilots = np.shape(pilot_symbs)[1]
    if numBlocks > numRefPilots:
        numBlocks = numRefPilots
        rec_symbs = rec_symbs[:,:int(numBlocks*pilot_ins_ratio)]
        rec_pilots = rec_pilots[:,:int(numBlocks)]
    elif numRefPilots > numBlocks:
        pilot_symbs = pilot_symbs[:,:int(numBlocks)]
    
    # Remove every X pilot symbol if selected
    if use_pilot_ratio >= pilot_symbs.shape[1]:
        raise ValueError("Can not use every %d pilots since only %d pilot symbols are present"%(use_pilot_ratio,pilot_symbs.shape[1]))
    rec_pilots = rec_pilots[:,::int(use_pilot_ratio)]
    pilot_symbs = pilot_symbs[:,::int(use_pilot_ratio)]
    
    # Check for a out of bounch error
    if pilot_symbs.shape[1] <= num_average:
        raise ValueError("Inpropper pilot symbol configuration. Larger averaging block size than total number of pilot symbols")
    
    # Should be an odd number to keey symmetry in averaging
    if not(num_average % 2):
        num_average += 1
    
    # Allocate output memory and process modes
    data_symbs = np.zeros([npols,np.shape(rec_symbs)[1]], dtype = complex)
    phase_trace = np.zeros([npols,np.shape(rec_symbs)[1]])
    res_phase = pilot_symbs.conjugate()*rec_pilots
    pilot_phase_base = np.unwrap(np.angle(res_phase), axis=-1)
    pilot_phase_average = filter.moving_average(pilot_phase_base,num_average)
    pp1 = pilot_phase_base[:, :int((num_average-1)/2)]
    pp2 = pilot_phase_average[:]
    t = pilot_phase_base[:,:int((num_average-1)/2)]
    t2 = pp2[:, -1].reshape(2,1)
    pp3 = np.ones(t.shape)*t2
    pilot_phase = np.hstack([pp1, pp2, pp3])
    pilot_pos = np.arange(0,pilot_phase.shape[-1]*pilot_ins_ratio*use_pilot_ratio, pilot_ins_ratio*use_pilot_ratio)
    pilot_pos_new = np.arange(0,pilot_phase.shape[-1]*pilot_ins_ratio*use_pilot_ratio)

    for l in range(npols):
        phase_trace[l,:] = np.interp(pilot_pos_new, pilot_pos, pilot_phase[l])
    data_symbs = rec_symbs*np.exp(-1j*phase_trace)

    # Remove the phase pilots after compensation. This is an option since they can be used for SNR estimation e.g.
    if remove_phase_pilots:
        pilot_pos = np.arange(0,np.shape(data_symbs)[1],pilot_ins_ratio)
        data_symbs = np.delete(data_symbs,pilot_pos, axis = 1)
        
    return data_symbs, phase_trace
    
    
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

def frame_sync(rx_signal, ref_symbs, os, frame_len=2 ** 16, M_pilot=4,
               mu=1e-3, Ntaps=17, **eqargs):
    """
    Locate the pilot sequence frame
    
    Uses a CMA-based search scheme to located the initiial pilot sequence in
    the long data frame. 

    Parameters
    ----------
    rx_signal: array_like
        Received Rx signal
    ref_symbs: array_like
        Pilot sequence
    os: int
        Oversampling factor
    frame_len: int
        Total frame length including pilots and payload
    M_pilot: int, optional
        QAM-order for pilot symbols
    mu: float, optional
        CMA step size
    Ntaps: int, optional
        number of taps to use in the equaliser
    **eqargs:
        arguments to be passed to equaliser

        
    Returns
    -------
    shift_factor: array_like
        location of frame start index per polarization
    foe_corse:  array_like
        corse frequency offset
    mode_sync_order: array_like
        Synced descrambled reference pattern order
    """
    rx_signal = np.atleast_2d(rx_signal)
    ref_symbs = np.atleast_2d(ref_symbs)
    pilot_seq_len = ref_symbs.shape[-1]
    nmodes = rx_signal.shape[0]
    assert rx_signal.shape[-1] >= (frame_len + 2*pilot_seq_len)*os, "Signal must be at least as long as frame"
    mode_sync_order = np.zeros(nmodes, dtype=int)
    not_found_modes = np.arange(0, nmodes)
    search_overlap = 2 # fraction of pilot_sequence to overlap
    step_width = pilot_seq_len * os
    step = step_width // search_overlap
    sig_len = rx_signal.shape[-1]
    # we only need to search the length of one frame*os plus some buffer
    num_steps = (frame_len*os)//step + search_overlap + 1
    # Now search for every mode independent
    shift_factor = np.zeros(nmodes, dtype=int)
    # Search based on equalizer error. Avoid one pilot_seq_len part in the beginning and
    # end to ensure that sufficient symbols can be used for the search
    sub_vars = np.ones((nmodes, num_steps)) * 1e2
    wxys = np.zeros((num_steps, nmodes, nmodes, Ntaps), dtype=complex)
    for i in np.arange(search_overlap, num_steps): # we avoid one step at the beginning
        tmp = rx_signal[:, i*step:i*step+step_width]
        wxy, err_out = equalisation.equalise_signal(tmp, os, mu, M_pilot, Ntaps=Ntaps, **eqargs)
        wxys[i] = wxy
        sub_vars[:,i] = np.var(err_out[:,int(-step/os+Ntaps):])
    # Lowest variance of the CMA error for each pol
    min_range = np.argmin(sub_vars, axis=-1)
    wxy = wxys[min_range]
    for l in range(nmodes):
        idx_min = min_range[l]
        # Extract a longer sequence to ensure that the complete pilot sequence is found
        longSeq = rx_signal[:, (idx_min+search_overlap)*step-step_width: (idx_min + search_overlap)*step+step_width]
        # Apply filter taps to the long sequence and remove coarse FO
        wx1 = wxy[l]
        symbs_out = equalisation.apply_filter(longSeq,os,wx1)
        foe_corse = phaserecovery.find_freq_offset(symbs_out)
        symbs_out = phaserecovery.comp_freq_offset(symbs_out, foe_corse)
        # Check for pi/2 ambiguties and verify all
        max_phase_rot = np.zeros(nmodes, dtype=np.float64)
        found_delay = np.zeros(nmodes, dtype=np.int)
        for ref_pol in not_found_modes:
            ix, dat, ii, ac = ber_functions.find_sequence_offset_complex(ref_symbs[ref_pol], symbs_out[l], show_cc=True)
            found_delay[ref_pol] = ix
            max_phase_rot[ref_pol] = ac
        # Check for which mode found and extract the reference delay
        max_sync_pol = np.argmax(max_phase_rot)
        mode_sync_order[l] = max_sync_pol
        symb_delay = found_delay[max_sync_pol]
        # Remove the found reference mode
        not_found_modes = not_found_modes[not_found_modes != max_sync_pol]
        # New starting sample
        shift_factor[l] = (idx_min + search_overlap)*step + os*symb_delay - step_width
    return shift_factor, foe_corse, mode_sync_order

def correct_shifts(shift_factors, ntaps, os):
    # taps cause offset on shift factors
    shift_factors = np.asarray(shift_factors)
    if not((ntaps[1]-ntaps[0])%os  ==  0):
        raise ValueError("Taps for search and convergence impropper configured")
    tap_cor = int((ntaps[1]-ntaps[0])/2)
    shift_factors -= tap_cor
    return shift_factors

def shift_signal(sig, shift_factors):
    k = len(shift_factors)
    if k > 1:
        for i in range(k):
            sig[i] = np.roll(sig[i], shift_factors[i])
    else:
        sig = np.roll(sig, shift_factors, axis=-1)
    return sig


def equalize_pilot_sequence(rx_signal, ref_symbs, os, shift_factor=0,  sh = False, process_frame_id = 0, frame_length = 2**16, mu = (1e-4,1e-4), M_pilot = 4, ntaps = (25,45), Niter = (10,30), adap_step = (True,True), method=('cma','cma')):
    """
    
    """
    # Inital settings
    rx_signal = np.atleast_2d(rx_signal)
    ref_symbs = np.atleast_2d(ref_symbs)
    npols = rx_signal.shape[0]    
    pilot_seq_len = ref_symbs.shape[-1]

    # Tap update and extract the propper pilot sequuence    

    # Run FOE and shift spectrum
    if sh:
       # Signal should be DC-centered. Do nothing.
       sig_dc_center = rx_signal
       foePerMode = np.zeros([npols,1])
    else:
        # First Eq, extract pilot sequence to do FOE
        pilot_seq = rx_signal[:,shift_factor:shift_factor+pilot_seq_len*os+ntaps[1]-1]
        wx, err = equalisation.equalise_signal(pilot_seq, os, mu[0], M_pilot,Ntaps = ntaps[1], Niter = Niter[1], method = method[0],adaptive_stepsize = adap_step[1])
        symbs_out= equalisation.apply_filter(pilot_seq,os,wx)

        # Pilot-based FOE
        foe, foePerMode, cond = pilot_based_foe(symbs_out, ref_symbs)
        foePerMode = np.ones(foePerMode.shape)*foe
        # Apply FO-compensation
        sig_dc_center = phaserecovery.comp_freq_offset(rx_signal, foePerMode, os=os)

    # First step Eq
    pilot_seq = sig_dc_center[:,shift_factor:+shift_factor+pilot_seq_len*os+ntaps[1]-1]
    wxy_init, err = equalisation.equalise_signal(pilot_seq, os, mu[1], 4, Ntaps = ntaps[1], Niter = Niter[1], method = method[0],adaptive_stepsize = adap_step[1])
    out_taps, err = equalisation.equalise_signal(pilot_seq, os, mu[1], 4,wxy=wxy_init,Ntaps = ntaps[1], Niter = Niter[1], method = method[1],adaptive_stepsize = adap_step[1])
    symbs_out = equalisation.apply_filter(pilot_seq,os,out_taps)

    # Equalize using additional frames if selected

    return out_taps, foePerMode


def find_const_phase_offset(rec_pilots, ref_symbs):
    """
    Finds a constant phase offset between the decoded pilot 
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
    """
    Corrects a constant phase offset between the decoded pilot 
    symbols and the transmitted ones
    
    Input:
        symbs: Complex symbols to be compensated
        phase_offsets: Phase offset for each mode
        
    Output:
        symbs: Symbols after phase rotation
    """
    
    symbs = np.atleast_2d(symbs)
    phase_offsets = np.atleast_2d(phase_offsets)
    npols = symbs.shape[0]

    for l in range(npols):
        symbs[l,:] = symbs[l,:] * np.exp(-1j*phase_offsets[l,0])

    return symbs

