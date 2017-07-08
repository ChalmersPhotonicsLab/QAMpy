# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 09:59:52 2017

@author: fotonik
"""

import numpy as np
from dsp import signals, equalisation, modulation, utils, phaserecovery, dsp_cython, signal_quality, ber_functions, pilotbased_receiver
from scipy.io import loadmat, savemat
import matplotlib.pylab as plt


def run_pilot_receiver_measdata(rec_signal, pilot_symbs, sys_config):

    os = sys_config['os']
    M = sys_config['M']
    Numtaps = sys_config['Numtaps']
    frame_length = sys_config['frame_length']
    pilot_seq_len = sys_config['pilot_seq_len']
    pilot_ins_ratio = sys_config['pilot_ins_ratio']
    
    
    # Fix the signal

    X = rec_signal[0,:]
    Y = rec_signal[1,:]
    
    X = X.flatten()
    Y = Y.flatten()
    
    X = utils.resample(X, 2.5, 2, renormalise = True)
    Y = utils.resample(Y, 2.5, 2, renormalise = True)
    X = utils.comp_IQbalance(X)
    Y = utils.comp_IQbalance(Y)
    
    delay = 70e4+18e3+256 # To play with
    
    SS = np.vstack([X[delay:],Y[delay:]])
    
    # Pilot symbols adapted for NeoPhotonics hybrid in Tx lab
    ref_symbs = (pilot_symbs[:,:pilot_seq_len])
    
    # Frame sync
    eq_pilots, shift_factor , taps, corse_foe = pilotbased_receiver.frame_sync(SS, ref_symbs, os, frame_length, mu=(1e-3,1e-3), ntaps=Numtaps,  Niter = (30,30), adap_step = (True,True))
    phase_offset = pilotbased_receiver.find_const_phase_offset(eq_pilots,ref_symbs)   
    eq_pilots = pilotbased_receiver.correct_const_phase_offset(eq_pilots,phase_offset)
    
    # Foe estimate
    foe, foePerMode, condNum = pilotbased_receiver.pilot_based_foe(eq_pilots,ref_symbs)
    corr_pilots = np.zeros(np.shape(eq_pilots),dtype=complex)
    #phase_offset = np.zeros(npols,dtype=float)
    for l in range(npols):
        # Remove FOE from pilots
        comp_test = phaserecovery.comp_freq_offset(eq_pilots[l,:],foePerMode[l]) 
        
        # Correct static phase difference between the Tx and Rx pilots
        #phase_offset[l] = pilotbased_receiver.find_const_phase_offset(comp_test,ref_symbs[l,:])    
        corr_pilots[l,:] = pilotbased_receiver.correct_const_phase_offset(comp_test,phase_offset[l])
    
    # Equalize the signal
    comp_test_sig = []
    for l in range(npols):
        test_sig = equalisation.apply_filter(SS[:,shift_factor[l]:shift_factor[l]+frame_length*os+Numtaps-1],os,taps[l])   
        test_sig_after_foe = phaserecovery.comp_freq_offset(test_sig[l,:],foePerMode[l])
        comp_test = pilotbased_receiver.correct_const_phase_offset(test_sig_after_foe, phase_offset[l])
        comp_test_sig.append(comp_test)
    
    # Pilot-aided CPE
    phase_comp_symbs = []
    phase_trace = []
    for l in range(npols):   
        symbs, trace = pilotbased_receiver.pilot_based_cpe(comp_test_sig[l][:,pilot_seq_len:], pilot_symbs[l,pilot_seq_len:], pilot_ins_ratio,use_pilot_ratio=1, num_average = 1, max_num_blocks = None)
        phase_comp_symbs.append(symbs)
        phase_trace.append(trace)

    #for l in range(npols):
    #    wx, err_both = equalisation.equalise_signal(phase_comp_symbs[l], 1,1e-3, M,Niter=4, Ntaps=Numtaps, method="sbd" , adaptive_stepsize=True)
    #    phase_trace[l] = equalisation.apply_filter(phase_comp_symbs[l], 1, wx)


    return eq_pilots, corr_pilots, phase_comp_symbs, phase_trace, shift_factor

def run_pilot_receiver_simulationdata(rec_signal, pilot_symbs, sys_config):

    
    os = sys_config['os']
    M = sys_config['M']
    Numtaps = sys_config['Numtaps']
    frame_length = sys_config['frame_length']
    pilot_seq_len = sys_config['pilot_seq_len']
    pilot_ins_ratio = sys_config['pilot_ins_ratio']
    
    
    if np.shape(rec_signal)[0] == 1:    
        rec_signal = np.vstack([rec_signal,rec_signal])
        pilot_symbs = np.vstack([pilot_symbs,pilot_symbs])
    
    npols = np.shape(rec_signal)[0]
    
    # Pilot symbols adapted for NeoPhotonics hybrid in Tx lab
    ref_symbs = (pilot_symbs[:,:pilot_seq_len])
    
    # Frame sync
    eq_pilots, shift_factor , taps, corse_foe = pilotbased_receiver.frame_sync(rec_signal, ref_symbs, os, frame_length, mu=(1e-3,1e-3), ntaps=Numtaps,  Niter = (30,30), adap_step = (True,True))
    phase_offset = pilotbased_receiver.find_const_phase_offset(eq_pilots,ref_symbs)   
    eq_pilots = pilotbased_receiver.correct_const_phase_offset(eq_pilots,phase_offset)
    
    # Foe estimate
    foe, foePerMode, condNum = pilotbased_receiver.pilot_based_foe(eq_pilots,ref_symbs)
    corr_pilots = np.zeros(np.shape(eq_pilots),dtype=complex)
    #phase_offset = np.zeros(npols,dtype=float)
    for l in range(npols):
        # Remove FOE from pilots
        comp_test = phaserecovery.comp_freq_offset(eq_pilots[l,:],foePerMode[l]) 
        
        # Correct static phase difference between the Tx and Rx pilots
        #phase_offset[l] = pilotbased_receiver.find_const_phase_offset(comp_test,ref_symbs[l,:])    
        corr_pilots[l,:] = pilotbased_receiver.correct_const_phase_offset(comp_test,phase_offset[l])
    
    # Equalize the signal
    comp_test_sig = []
    for l in range(npols):
        test_sig = equalisation.apply_filter(rec_signal[:,shift_factor[l]:shift_factor[l]+frame_length*os+Numtaps-1],os,taps[l])   
        test_sig_after_foe = phaserecovery.comp_freq_offset(test_sig[l,:],foePerMode[l])
        comp_test = pilotbased_receiver.correct_const_phase_offset(test_sig_after_foe, phase_offset[l])
        comp_test_sig.append(comp_test)
    
    # Pilot-aided CPE
    phase_comp_symbs = []
    phase_trace = []
    for l in range(npols):   
        symbs, trace = pilotbased_receiver.pilot_based_cpe(comp_test_sig[l][:,pilot_seq_len:], pilot_symbs[l,pilot_seq_len:], pilot_ins_ratio,use_pilot_ratio=1, num_average = 1, max_num_blocks = None)
        phase_comp_symbs.append(symbs)
        phase_trace.append(trace)

    #for l in range(npols):
    #    wx, err_both = equalisation.equalise_signal(phase_comp_symbs[l], 1,1e-3, M,Niter=4, Ntaps=Numtaps, method="sbd" , adaptive_stepsize=True)
    #    phase_trace[l] = equalisation.apply_filter(phase_comp_symbs[l], 1, wx)


    return eq_pilots, corr_pilots, phase_comp_symbs, phase_trace, shift_factor



# Tx Config

sys_config = {'os':2,'M':128, 'frame_length':2**16, 'pilot_seq_len':256, \
              'pilot_ins_ratio':32,'Numtaps':45 }




rec_signal = meas['Meas2']
pilot_symbs = res['pilot_symbs']
pilot_symbs = -np.conj(np.vstack([(pilot_symbs),pilot_symbs])) # NeoPhotonics Hybrid
data_symbs = -np.conj(res['data_symbs'])


# Call and run the equalizer
eq_pilots, corr_pilots, symbs_out, phase_trace, shift_factor = run_pilot_receiver_measdata(rec_signal, pilot_symbs, sys_config)



# Verify stuff rot the result
QAM = modulation.QAMModulator(sys_config[M])
serX = QAM.calculate_SER(symbs_out[0][0,:-1], symbol_tx=data_symbs[0,:])[0]
serY = QAM.calculate_SER(symbs_out[1][0,:-1], symbol_tx=data_symbs[0,:])[0]

berX, errsx, seqLengthx = QAM.cal_BER(symbs_out[0][0,:-1], syms_tx = data_symbs[0,:])
berY, errsy, seqLengthy = QAM.cal_BER(symbs_out[1][0,:-1], syms_tx = data_symbs[0,:])


#  Verification and plotting    
plt.figure()
plt.subplot(221)
plt.plot(eq_pilots[0,:].real,eq_pilots[0,:].imag,'.')
plt.title('Pilots after Eq X-Pol')  
plt.subplot(222)
plt.plot(eq_pilots[1,:].real,eq_pilots[1,:].imag,'.')
plt.title('Pilots after Eq Y-Pol')  
#plt.plot(comp_test[0,:].real,comp_test[0,:].imag,'.')  
#plt.title('Pilots after FOE')
plt.subplot(223)
plt.plot(corr_pilots[0,:].real,corr_pilots[0,:].imag,'.')  
plt.title('Phase-corrected pilots X-Pol')
plt.subplot(224)
plt.plot(corr_pilots[1,:].real,corr_pilots[1,:].imag,'.')  
plt.title('Phase-corrected pilots X-Pol')

# Plot constellation
plt.figure()
plt.hexbin(symbs_out[0][0,:].real, symbs_out[0][0,:].imag)
plt.title('Pilot-based X-Pol: EVM %2.2f%%'%(QAM.cal_EVM(symbs_out[0][0,:])*100))

plt.figure()
plt.hexbin(symbs_out[1][0,:].real, symbs_out[1][0,:].imag)
plt.title('Pilot-based Y-Pol: EVM %2.2f%%'%(QAM.cal_EVM(symbs_out[1][0,:])*100))

# Only verification stuff for BPS
#
bps_out_x = phaserecovery.blindphasesearch(symbs_out[0][0,:],64,QAM.symbols, 128)
bps_out_y = phaserecovery.blindphasesearch(symbs_out[1][0,:],64,QAM.symbols, 128)
plt.figure()
plt.hexbin(bps_out_x[0].real, bps_out_x[0].imag)
plt.title('BPS X-POL:  EVM %2.2f%%'%(QAM.cal_EVM(bps_out_x[0])*100))
plt.figure()
plt.hexbin(bps_out_y[0].real, bps_out_y[0].imag)
plt.title('BPS X-POL:  EVM %2.2f%%'%(QAM.cal_EVM(bps_out_y[0])*100))