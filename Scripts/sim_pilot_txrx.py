import numpy as np
from dsp import equalisation, modulation, phaserecovery, pilotbased_receiver
from qampy.core import pilotbased_transmitter


def run_pilot_receiver(rec_signal, pilot_symbs, process_frame_id=0, sh=False, os=2, M=128, Numtaps=(17, 45),
                       frame_length=2 ** 16, method=('cma', 'cma'), pilot_seq_len=512, pilot_ins_ratio=32,
                       Niter=(10, 30), mu=(1e-3, 1e-3), adap_step=(True, True), cpe_average=5, use_cpe_pilot_ratio=1,
                       remove_inital_cpe_output=True, remove_phase_pilots=True):

    rec_signal = np.atleast_2d(rec_signal)
    tap_cor = int((Numtaps[1] - Numtaps[0]) / 2)
    npols = rec_signal.shape[0]
    # Extract pilot sequence
    ref_symbs = (pilot_symbs[:, :pilot_seq_len])

    # Frame sync, locate first frame
    shift_factor, corse_foe, mode_alignemnt = pilotbased_receiver.frame_sync(rec_signal, ref_symbs, os, frame_length=frame_length,
                                                             mu=mu[0], method=method[0], ntaps=Numtaps[0],
                                                             Niter=Niter[0], adap_step=adap_step[0])
    
    # Redistribute pilots according to found modes
    pilot_symbs = pilot_symbs[mode_alignemnt,:]
    ref_symbs = ref_symbs[mode_alignemnt,:]
    
    # Converge equalizer using the pilot sequence
    eq_pilots, foePerMode, taps, shift_factor = pilotbased_receiver.equalize_pilot_sequence(rec_signal, ref_symbs,
                                                                                            shift_factor, os, sh=sh,
                                                                                            process_frame_id=process_frame_id,
                                                                                            frame_length=frame_length,
                                                                                            mu=mu, method=method,
                                                                                            ntaps=Numtaps, Niter=Niter,
                                                                                            adap_step=adap_step)

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

# Standard function to test DSP
def sim_pilot_txrx(sig_snr, Ntaps=45, beta=0.1, M=64, freq_off = None,cpe_avg=8,
                   frame_length = 2**14, pilot_seq_len = 8192, pilot_ins_rat=32,
                   num_frames=3,modal_delay=None, laser_lw = None, 
                   resBits_tx=None, resBits_rx=None):
    
    npols=2
    
    # Create frame
    frame, data_symbs, pilot_symbs = pilotbased_transmitter.gen_dataframe_with_phasepilots(M, npols, frame_length=frame_length,
                                                                                           pilot_seq_len=pilot_seq_len,
                                                                                           pilot_ins_ratio=pilot_ins_rat)
    
    # Simulate transmission
    sig_tx = pilotbased_transmitter.sim_tx(frame, 2, snr=sig_snr, modal_delay=None, freqoff=freq_off,
                                           linewidth=laser_lw, beta=beta, num_frames=3, resBits_tx=resBits_tx,
                                           resBits_rx=resBits_rx)
    
    
    # Run DSP
    dsp_out= run_pilot_receiver(sig_tx,pilot_symbs, 
                                 frame_length=frame_length, M=M, pilot_seq_len=pilot_seq_len, 
                                 pilot_ins_ratio=pilot_ins_rat,cpe_average=cpe_avg,os= 2,
                                 Numtaps=(17,Ntaps), mu = (1e-3, 1e-3), method=("cma","sbd"))

    # Calculate GMI and BER
    mod = modulation.QAMModulator(M)
    gmi_res = np.zeros(npols)
    ber_res = np.zeros(npols)
    for l in range(npols):
        gmi_res[l] = mod.cal_gmi(dsp_out[1][l].flatten(),data_symbs[l])[0][0]
        ber_res[l] = mod.cal_ber(dsp_out[1][l].flatten(),symbols_tx = data_symbs[l])
        
    return gmi_res, ber_res
