import numpy as np
from dsp import equalisation, modulation, utils, phaserecovery, pilotbased_receiver,pilotbased_transmitter,filter,\
    resample,impairments
import matplotlib.pylab as plt
import copy

def run_pilot_receiver(rec_signal, pilot_symbs, process_frame_id=0, sh=False, os=2, M=128, Numtaps=(17, 45),
                       frame_length=2 ** 16, method=('cma', 'cma'), pilot_seq_len=512, pilot_ins_ratio=32,
                       Niter=(10, 30), mu=(1e-3, 1e-3), adap_step=(True, True), cpe_average=5, use_cpe_pilot_ratio=1,
                       remove_inital_cpe_output=True, remove_phase_pilots=True, do_pilot_based_foe=True,
                       foe_symbs=None, blind_foe_payload=False):

    rec_signal = np.atleast_2d(rec_signal)
    tap_cor = int((Numtaps[1] - Numtaps[0]) / 2)
    npols = rec_signal.shape[0]
    # Extract pilot sequence
    ref_symbs = (pilot_symbs[:, :pilot_seq_len])

    # Frame sync, locate first frame
    shift_factor, corse_foe = pilotbased_receiver.frame_sync(rec_signal, ref_symbs, os, frame_length=frame_length,
                                                             mu=mu[0], method=method[0], ntaps=Numtaps[0],
                                                             Niter=Niter[0], adap_step=adap_step[0])

    # Converge equalizer using the pilot sequence
    eq_pilots, foePerMode, taps, shift_factor = pilotbased_receiver.equalize_pilot_sequence(rec_signal, ref_symbs,
                                                                                            shift_factor, os, sh=sh,
                                                                                            process_frame_id=process_frame_id,
                                                                                            frame_length=frame_length,
                                                                                            mu=mu, method=method,
                                                                                            ntaps=Numtaps, Niter=Niter,
                                                                                            adap_step=adap_step,
                                                                                            do_pilot_based_foe=do_pilot_based_foe,
                                                                                            foe_symbs=foe_symbs,
                                                                                            blind_foe_payload=blind_foe_payload)

    # DSP for the payload: Equalization, FOE, CPE. All pilot-aided
    dsp_sig_out = []
    phase_trace = []
    for l in range(npols):
        # Extract syncrhonized sequenceM, symbs_out, data_symbs
        mode_sig = rec_signal[:,
                   shift_factor[l] - tap_cor:shift_factor[l] - tap_cor + frame_length * os + Numtaps[1] - 1]

        # In non-SH operation do FOE
        if not sh:
            mode_sig = phaserecovery.comp_freq_offset(mode_sig, foePerMode, os=os)

        # Equalization of signal
        eq_mode_sig = equalisation.apply_filter(mode_sig, os, taps[l])

        # Pilot-aided CPE
        symbs, trace = pilotbased_receiver.pilot_based_cpe(eq_mode_sig[l, pilot_seq_len:],
                                                           pilot_symbs[l, pilot_seq_len:], pilot_ins_ratio,
                                                           use_pilot_ratio=use_cpe_pilot_ratio, num_average=cpe_average,
                                                           remove_phase_pilots=remove_phase_pilots)

        if remove_inital_cpe_output:
            symbs = symbs[:, int(use_cpe_pilot_ratio * pilot_ins_ratio * cpe_average / 2):-int(
                use_cpe_pilot_ratio * pilot_ins_ratio * cpe_average / 2)]
            # symbs = symbs[:, int(use_cpe_pilot_ratio * pilot_ins_ratio * cpe_average):]
        dsp_sig_out.append(symbs)
        phase_trace.append(trace)

    return eq_pilots, dsp_sig_out, shift_factor, taps, phase_trace, foePerMode


def run_joint_pilot_receiver(rec_signal, pilot_symbs, process_frame_id=0, sh=False, os=2, M=128, Numtaps=(17, 45),
                       frame_length=2**16, method=('cma', 'cma'), pilot_seq_len=512, pilot_ins_ratio=32,
                       Niter=(10, 30), mu=(1e-3, 1e-3), adap_step=(True, True), cpe_average=5, use_cpe_pilot_ratio=1,
                       remove_inital_cpe_output=True, remove_phase_pilots=True, do_pilot_based_foe=True,
                       foe_symbs=None, blind_foe_payload=False,ch_sep=2):

    # Check for propper dim
    if not len(rec_signal) == 3:
        raise ValueError("Requires an input signal array with 3 neighboring channels")
        
    if not len(pilot_symbs) == 3:
        raise ValueError("Requires an input pilot symbol array with length 3")
    

    tap_cor = int((Numtaps[1] - Numtaps[0]) / 2)
    npols = rec_signal[1].shape[0]
    
    # Extract pilot sequence
    ref_symbs = (pilot_symbs[1][:, :pilot_seq_len])
    
    ref_symbs_wdm = [pilot_symbs[0][:, :pilot_seq_len],pilot_symbs[1][:, :pilot_seq_len],pilot_symbs[2][:, :pilot_seq_len]]
    
    # Frame sync, locate first frame
    shift_factor, corse_foe = pilotbased_receiver.frame_sync(rec_signal[1], ref_symbs, os, frame_length=frame_length,
                                                             mu=mu[0], method=method[0], ntaps=Numtaps[0],
                                                             Niter=Niter[0], adap_step=adap_step[0])  
    


    # Converge equalizer using the pilot sequence
    eq_pilots, foePerMode, taps, shift_factor, out_sig, data_out = pilotbased_receiver.equalize_pilot_sequence_joint(rec_signal, ref_symbs_wdm,
                                                                                            shift_factor, os, sh=sh,
                                                                                            process_frame_id=process_frame_id,
                                                                                            frame_length=frame_length,
                                                                                            mu=mu, method=method,
                                                                                            ntaps=Numtaps, Niter=Niter,
                                                                                            adap_step=adap_step,
                                                                                            do_pilot_based_foe=do_pilot_based_foe,
                                                                                            foe_symbs=foe_symbs,
                                                                                            blind_foe_payload=blind_foe_payload,ch_sep=ch_sep)




    sig_dc_low = rec_signal[0]
    sig_dc_cent = rec_signal[1]
    sig_dc_high = rec_signal[2]
    
    
    sig_dc_low *= np.exp(-1j*2*np.pi*ch_sep*np.linspace(0,sig_dc_low.shape[1]/os,sig_dc_low.shape[1]))
    

    sig_dc_high *= np.exp(1j*2*np.pi*ch_sep*np.linspace(0,sig_dc_high.shape[1]/os,sig_dc_high.shape[1]))


    mode_sig = np.concatenate([sig_dc_low,sig_dc_cent,sig_dc_high],axis=0)

    # DSP for the payload: Equalization, FOE, CPE. All pilot-aided
    dsp_sig_out = []
    phase_trace = []
    for l in range(npols):
        # Extract syncrhonized sequenceM, symbs_out, data_symbs
        rx_mode_sig = mode_sig[:,shift_factor[l] - tap_cor:shift_factor[l] - tap_cor + frame_length * os + Numtaps[1] - 1]


        # Equalization of signal
        eq_mode_sig = equalisation.apply_filter(rx_mode_sig, os, taps[l])    

        
        # Pilot-aided CPE
        symbs, trace = pilotbased_receiver.pilot_based_cpe(eq_mode_sig[2+l,pilot_seq_len:],
                                                           pilot_symbs[1][l, pilot_seq_len:], pilot_ins_ratio,
                                                           use_pilot_ratio=use_cpe_pilot_ratio, num_average=cpe_average,
                                                           remove_phase_pilots=remove_phase_pilots)

        if remove_inital_cpe_output:
            symbs = symbs[:, int(use_cpe_pilot_ratio * pilot_ins_ratio * cpe_average / 2):-int(
                use_cpe_pilot_ratio * pilot_ins_ratio * cpe_average / 2)]
            # symbs = symbs[:, int(use_cpe_pilot_ratio * pilot_ins_ratio * cpe_average):]
        dsp_sig_out.append(symbs)
        phase_trace.append(trace)

    return eq_pilots, dsp_sig_out, shift_factor, taps, phase_trace, foePerMode


def pre_filter(signal, bw, os,center_freq = 0):
    """
    Low-pass pre-filter signal with square shape filter

    Parameters
    ----------

    signal : array_like
        single polarization signal

    bw     : float
        bandwidth of the rejected part, given as fraction of overall length
    """
    N = len(signal)
    freq_axis = np.fft.fftfreq(N, 1 / os)

    idx = np.where(abs(freq_axis-center_freq) < bw / 2)

    h = np.zeros(N, dtype=np.float64)
    # h[int(N/(bw/2)):-int(N/(bw/2))] = 1
    h[idx] = 1
    s = np.fft.ifftshift(np.fft.ifft(np.fft.fft(signal) * h))
    return s







def sim_sep(ch_sep,rx_filter_bw = 1.2,beta=0.1,sig_snr=35):

    # Select Rx channel
    sel_wdm_ch = np.array([-1,0,1])
    sel_ref_ch = 1
    # If wanted, plot the output results. 
    plot_results = False


    
    
    # Over sampling settings and WDM configuration
    os_tx = 8
    #rx_filter_bw = 1.1
    n_wdm_channels = 3
    #ch_sep = 1.000
    os_rx = 2

    # Signal configuration
    M = 64
    npols = 2
#    beta = 0.1
    laser_lw = 10e3
    #sig_snr = 35
    freq_off = 200e6*0
    # Pilot DSP configuration
    frame_length = 2**14
    pilot_seq_len = 8192
    pilot_ins_rat = 32
    cpe_avg = 8

    
    
    sig_wdm_ch = []
    frame_symbs = []
    data_symbs = []
    pilot_symbs = []

    #==============================================================================
    # Generate several frames
    #==============================================================================
    for ch in range(n_wdm_channels):
        # Create frame
        frames, datas, pilots = pilotbased_transmitter.gen_dataframe_with_phasepilots(M,npols,frame_length=frame_length,
                                                                                      pilot_seq_len=pilot_seq_len,
                                                                                      pilot_ins_ratio=pilot_ins_rat)
    
        # Add output vars
        frame_symbs.append(frames)
        data_symbs.append(datas)
        pilot_symbs.append(pilots)
    
        # Simulate tyransmission
    
        sig_tmp = pilotbased_transmitter.sim_tx(frame_symbs[ch], os_tx, snr=sig_snr, modal_delay=None, freqoff=freq_off,
                                                        linewidth=laser_lw,beta=beta)
           
        # Add signal to Rx structure. 
        sig_wdm_ch.append(sig_tmp)
    
    
    #==============================================================================
    # Place the data channels
    #==============================================================================
    num_side_ch = int((n_wdm_channels-1)/2)
    tx_sig = np.zeros(sig_wdm_ch[0].shape,dtype=complex)
    for ch in range(n_wdm_channels):
        tx_sig += sig_wdm_ch[ch] * np.exp(1j*2*np.pi*(-num_side_ch+ch)*ch_sep*
                            np.linspace(0,sig_wdm_ch[ch].shape[1]/os_tx,sig_wdm_ch[ch].shape[1]))
        print("Shitfactor_tx: %d"%(-num_side_ch+ch))
    

    rx_sig = copy.deepcopy(tx_sig)
    
    #==============================================================================
    # Filter a bit, select propper channel
    #==============================================================================
    
    rx_wdm_sigs = []
    for sel_ch in sel_wdm_ch:
        # Filter out per polarization
        print("Shiftfactor rx: %2.2f"%(sel_ch*ch_sep))
        rx_sig_ch = copy.deepcopy(rx_sig)
        for l in range(npols):
            #rx_sig_ch[l] = pre_filter(rx_sig_ch[l],rx_filter_bw,os_tx,center_freq=sel_ch*ch_sep)
            if (sel_ch) != 0:               
               rx_sig_ch[l] *= np.exp(-1j*2*np.pi*sel_ch*ch_sep*np.linspace(0,rx_sig.shape[1]/os_tx,rx_sig.shape[1]))
               rx_sig_ch[l] = pre_filter(rx_sig_ch[l],rx_filter_bw/2,os_tx,center_freq=sel_ch*ch_sep/2)
            else:
               rx_sig_ch[l] = pre_filter(rx_sig_ch[l],rx_filter_bw,os_tx,center_freq=sel_ch*ch_sep)
               
        # Add down-shifted signals to the output array
        rx_wdm_sigs.append(rx_sig_ch)
    
    #==============================================================================
    # Start the receiver structure
    #==============================================================================
    
    rx_wdm_sigs_resample = []
    for wdm_ch in range(len(rx_wdm_sigs)):
        
        # Orthonormalize a bit, try to get back to something again
        rx_sig = utils.orthonormalize_signal(rx_wdm_sigs[wdm_ch])
        
        # Resample for DSP
        rx_x = resample.rrcos_resample_zeroins(rx_sig[0].flatten(), os_tx, os_rx, Ts=1/os_rx, beta=beta, renormalise = True)
        rx_y = resample.rrcos_resample_zeroins(rx_sig[1].flatten(), os_tx, os_rx, Ts=1/os_rx, beta=beta, renormalise = True)
        rx_wdm_sigs_resample.append(np.vstack((rx_x,rx_y)))
    
#        rx_x = resample.resample_poly(rx_sig[0].flatten(), os_tx, os_rx, renormalise = True)
#        rx_y = resample.resample_poly(rx_sig[1].flatten(), os_tx, os_rx, renormalise = True)
#        rx_wdm_sigs_resample.append(np.vstack((rx_x,rx_y)))
    
    #==============================================================================
    # Run DSP and calculate AIR
    #==============================================================================
    
    # Run DSP
    dsp_out_single = run_pilot_receiver(rx_wdm_sigs_resample[sel_ref_ch],pilot_symbs[sel_ref_ch], 
                                 frame_length=frame_length, M=M, pilot_seq_len=pilot_seq_len, 
                                 pilot_ins_ratio=pilot_ins_rat,cpe_average=cpe_avg,os= os_rx)
    
    dsp_out = run_joint_pilot_receiver(rx_wdm_sigs_resample,pilot_symbs, 
                                 frame_length=frame_length, M=M, pilot_seq_len=pilot_seq_len, 
                                 pilot_ins_ratio=pilot_ins_rat,cpe_average=cpe_avg,os= os_rx,ch_sep=ch_sep)
    
    # Calculate GMI
    mod = modulation.QAMModulator(M)
    gmi_res = np.zeros(npols)
    gmi_res_single = np.zeros(npols)
    ber_res = np.zeros(npols)
    ber_res_single = np.zeros(npols)
    for l in range(npols):
        gmi_res[l] = mod.cal_gmi(dsp_out[1][l].flatten(),data_symbs[1][l])[0][0]
        gmi_res_single[l] = mod.cal_gmi(dsp_out_single[1][l].flatten(),data_symbs[sel_ref_ch][l])[0][0]
        ber_res[l] = mod.cal_ber(dsp_out[1][l].flatten(),symbols_tx = data_symbs[1][l])
        ber_res_single[l] = mod.cal_ber(dsp_out_single[1][l].flatten(),symbols_tx = data_symbs[sel_ref_ch][l])
    
    
    if plot_results:
        plt.figure()
        xax = np.fft.fftshift(np.fft.fftfreq(tx_sig[0].shape[0],1/os_tx))
        plt.semilogy(xax, np.fft.fftshift(np.abs(np.fft.fft(tx_sig[0]))**2),label="Tx Spectrum")
        plt.semilogy(xax, np.fft.fftshift(np.abs(np.fft.fft(rx_wdm_sigs[0][1]))**2),label="Left")
        plt.semilogy(xax, np.fft.fftshift(np.abs(np.fft.fft(rx_wdm_sigs[1][1]))**2),label="Center")
        plt.semilogy(xax, np.fft.fftshift(np.abs(np.fft.fft(rx_wdm_sigs[2][1]))**2),label="Right")
        plt.legend(frameon=True)
        plt.figure()
        plt.hist2d(dsp_out[1][0].flatten().real,dsp_out[1][0].flatten().imag,bins=200)
    
        plt.figure()
        plt.title("Pilots out test")
        plt.plot(dsp_out[0][0].real,dsp_out[0][0].imag,'.')
        plt.plot(dsp_out[0][1].real,dsp_out[0][1].imag,'.')
    
    return gmi_res, gmi_res_single, ber_res, ber_res_single
    
#==============================================================================
# Do Some plotting
#==============================================================================
#
#
#ch_sep_array = np.arange(1.3,0.8,-.02)
#num_avg = 3
#
#
#snr_test = np.arange(40,15,-2)
#beta_test =  np.arange(0.01,.5,.02)
#beta_test = np.array([0.01,0.03,0.05,0.1,0.2,0.5])
#gmi_res_joint = np.zeros([snr_test.shape[0],beta_test.shape[0]])
#gmi_res_single= np.zeros([snr_test.shape[0],beta_test.shape[0]])
#ber_res_joint = np.zeros([snr_test.shape[0],beta_test.shape[0]])
#ber_res_single= np.zeros([snr_test.shape[0],beta_test.shape[0]])
#
#for s in range(snr_test.shape[0]):
#    for b in range(beta_test.shape[0]):
#        gmi_joint_tmp = np.zeros(num_avg)
#        gmi_single_tmp = np.zeros(num_avg)
#        for n in range(num_avg):
#            print("SNR: %d, Beta: %1.2f, Avg. Ind: %d"%(snr_test[s],beta_test[b],n))
#            gmi1,gmi2,ber1,ber2 = sim_sep(1.0, beta=beta_test[b],sig_snr=snr_test[s])
#            gmi_joint_tmp[n] = np.sum(gmi1)
#            gmi_single_tmp[n] = np.sum(gmi2)
#            ber_res_joint[n] = np.mean(ber1)
#            ber_res_single[n] = np.mean(ber2)
#        gmi_res_joint[s,b] = np.mean(gmi_joint_tmp)
#        gmi_res_single[s,b] = np.mean(gmi_single_tmp)
#        ber_res_joint[s,b] = np.mean(ber_res_joint)
#        ber_res_single[s,b] = np.mean(ber_res_single)
#        print("SNR: %d, Beta: %1.2f, GMI-Joint: %2.2f, GMI-Ind. %2.2f"%
#              (snr_test[s],beta_test[b],gmi_res_joint[s,b],gmi_res_single[s,b]))
#
#np.savez("SNR_Beta_SimTest",snr_test=snr_test,beta_test=beta_test,gmi_res_joint=gmi_res_joint,
#         gmi_res_single=gmi_res_single,ber_res_joint=ber_res_joint,ber_res_single=ber_res_single)
#gmi_joint_0p8 = np.zeros([ch_sep_array.shape[0],num_avg])
#gmi_single_0p8 = np.zeros([ch_sep_array.shape[0],num_avg])
#
#for ind,curr_ch_sep in enumerate(ch_sep_array):
#    for avg_ind in range(num_avg):
#        batch_gmi_joint, batch_gmi_single = sim_sep(curr_ch_sep,beta=0.8)
#        gmi_joint_0p8[ind,avg_ind] = np.sum(batch_gmi_joint)
#        gmi_single_0p8[ind,avg_ind] = np.sum(batch_gmi_single)
#    plt.close('all')


#gmi_joint_0p5 = np.zeros([ch_sep_array.shape[0],num_avg])
#gmi_single_0p5 = np.zeros([ch_sep_array.shape[0],num_avg])
#
#for ind,curr_ch_sep in enumerate(ch_sep_array):
#    for avg_ind in range(num_avg):
#        batch_gmi_joint, batch_gmi_single = sim_sep(curr_ch_sep,beta=0.5)
#        gmi_joint_0p5[ind,avg_ind] = np.sum(batch_gmi_joint)
#        gmi_single_0p5[ind,avg_ind] = np.sum(batch_gmi_single)
#    plt.close('all')
#
#
#
#gmi_joint_0p2 = np.zeros([ch_sep_array.shape[0],num_avg])
#gmi_single_0p2 = np.zeros([ch_sep_array.shape[0],num_avg])
#for ind,curr_ch_sep in enumerate(ch_sep_array):
#    for avg_ind in range(num_avg):
#        batch_gmi_joint, batch_gmi_single = sim_sep(curr_ch_sep,beta=0.2)
#        gmi_joint_0p2[ind,avg_ind] = np.sum(batch_gmi_joint)
#        gmi_single_0p2[ind,avg_ind] = np.sum(batch_gmi_single)
#    plt.close('all')
#
#
#gmi_joint_0p1 = np.zeros([ch_sep_array.shape[0],num_avg])
#gmi_single_0p1 = np.zeros([ch_sep_array.shape[0],num_avg])
#
#for ind,curr_ch_sep in enumerate(ch_sep_array):
#    for avg_ind in range(num_avg):
#        batch_gmi_joint, batch_gmi_single = sim_sep(curr_ch_sep,beta=0.1)
#        gmi_joint_0p1[ind,avg_ind] = np.sum(batch_gmi_joint)
#        gmi_single_0p1[ind,avg_ind] = np.sum(batch_gmi_single)
#    plt.close('all')
        
#gmi_joint_0p05 = np.zeros([ch_sep_array.shape[0],num_avg])
#gmi_single_0p05 = np.zeros([ch_sep_array.shape[0],num_avg])
#
#for ind,curr_ch_sep in enumerate(ch_sep_array):
#    for avg_ind in range(num_avg):
#        batch_gmi_joint, batch_gmi_single = sim_sep(curr_ch_sep,beta=0.05)
#        gmi_joint_0p05[ind,avg_ind] = np.sum(batch_gmi_joint)
#        gmi_single_0p05[ind,avg_ind] = np.sum(batch_gmi_single)
#    plt.close('all')
#
#
#gmi_joint_0p01 = np.zeros([ch_sep_array.shape[0],num_avg])
#gmi_single_0p01 = np.zeros([ch_sep_array.shape[0],num_avg])
#for ind,curr_ch_sep in enumerate(ch_sep_array):
#    for avg_ind in range(num_avg):
#        batch_gmi_joint, batch_gmi_single = sim_sep(curr_ch_sep,beta=0.01)
#        gmi_joint_0p01[ind,avg_ind] = np.sum(batch_gmi_joint)
#        gmi_single_0p01[ind,avg_ind] = np.sum(batch_gmi_single)
#    plt.close('all')
    
    
#plt.figure()
#plt.plot()