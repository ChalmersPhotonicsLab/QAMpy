import numpy as np
from qampy.core import equalisation,  phaserecovery, pilotbased_receiver,pilotbased_transmitter,filter,\
    resample
from qampy import signals, impairments, helpers, phaserec
from qampy.equalisation import pilot_equaliser
import matplotlib.pylab as plt

def run_pilot_receiver2(rec_signal, process_frame_id=0, foe_comp=True, os=2, M=128, Numtaps=(17, 45),
                       frame_length=2 ** 16, method=('cma', 'cma'), pilot_seq_len=512, pilot_ins_ratio=32,
                       Niter=(10, 30), mu=(1e-3, 1e-3), adap_step=(True, True), cpe_average=5, use_cpe_pilot_ratio=1,
                       remove_inital_cpe_output=True, remove_phase_pilots=True, nframes=1):
    rec_signal.sync2frame(Ntaps=Numtaps[0], mu=mu[0], method=method[0], adaptive_stepsize=adap_step[0])
    #shift_factor = pilotbased_receiver.correct_shifts(rec_signal.shiftfctrs, Numtaps, rec_signal.os)
    #signal = np.roll(signal, -shift_factor[shift_factor>=0].min(), axis=-1)
    #shift_factors -= shift_factors[shift_factors>=0].min()
    #signal.shiftfctrs = shift_factors
    taps_all, eq_mode_sig = pilot_equaliser(rec_signal, mu, Numtaps[1], apply=True, foe_comp=foe_comp, adaptive_stepsize=adap_step[1],
                                            Niter=Niter[1], methods=method)
    #symbs, trace = pilotbased_receiver.pilot_based_cpe(eq_mode_sig[:, eq_mode_sig._pilot_seq_len:eq_mode_sig.frame_len],
                                                           #eq_mode_sig.ph_pilots, eq_mode_sig._pilot_ins_rat,
                                                              #use_pilot_ratio=use_cpe_pilot_ratio, num_average=cpe_average,
                                                       #remove_phase_pilots=True)
    symbs, trace = phaserec.pilot_cpe(eq_mode_sig, N=cpe_average, pilot_rat=use_cpe_pilot_ratio, nframes=1, use_seq=True)
    #symbs = eq_mode_sig
    return symbs, trace, eq_mode_sig, taps_all

def run_pilot_receiver(rec_signal, process_frame_id=0, foe_comp=True, os=2, M=128, Numtaps=(17, 45),
                       frame_length=2 ** 16, method=('cma', 'cma'), pilot_seq_len=512, pilot_ins_ratio=32,
                       Niter=(10, 30), mu=(1e-3, 1e-3), adap_step=(True, True), cpe_average=5, use_cpe_pilot_ratio=1,
                       remove_inital_cpe_output=True, remove_phase_pilots=True):
    ref_symbs = rec_signal.pilot_seq
    nmodes = rec_signal.shape[0]

    # Frame sync, locate first frame
    shift_factor, corse_foe, mode_alignment, wx1 = pilotbased_receiver.frame_sync(rec_signal, ref_symbs, os, frame_len=frame_length,
                                                                             mu=mu[0], method=method[0], Ntaps=Numtaps[0],
                                                                                                                adaptive_stepsize=adap_step[0])

    # Redistribute pilots according to found modes
    pilot_symbs = rec_signal.pilots[mode_alignment,:]
    ref_symbs = ref_symbs[mode_alignment,:]
    # taps cause offset on shift factors
    shift_factor = pilotbased_receiver.correct_shifts(shift_factor, Numtaps, os)
    # shift so that modes are offset from minimum mode minimum mode is align with the frame
    rec_signal = np.roll(rec_signal, -shift_factor[shift_factor>=0].min(), axis=-1)
    shift_factor -= shift_factor[shift_factor>=0].min()


    # Converge equalizer using the pilot sequence
    taps_all = []
    foe_all = []
    if np.all(shift_factor == 0):
        taps_all, foe_all = pilotbased_receiver.equalize_pilot_sequence(rec_signal, ref_symbs, os, foe_comp=foe_comp, mu=mu,
                                                                        Ntaps=Numtaps[1], Niter=Niter[1],
                                                                        adaptive_stepsize=adap_step[1], methods=method)
    else:
        for i in range(nmodes):
            rec_signal2 = np.roll(rec_signal, -shift_factor[i], axis=-1)
            taps, foePerMode = pilotbased_receiver.equalize_pilot_sequence(rec_signal2, ref_symbs, os, foe_comp=foe_comp, mu=mu,
                                                                           Ntaps=Numtaps[1], Niter=Niter[1],
                                                                           adaptive_stepsize=adap_step[1],
                                                                           methods=method)
            taps_all.append(taps[i])
            foe_all.append(foePerMode[i])

    if foe_comp:
        out_sig = phaserecovery.comp_freq_offset(rec_signal, np.array(foe_all), os=os)
        out_sig = rec_signal.recreate_from_np_array(out_sig)
    else:
        out_sig = rec_signal
    eq_mode_sig1 = pilotbased_receiver.shift_signal(out_sig, shift_factor)
    eq_mode_sig = equalisation.apply_filter(eq_mode_sig1, os, np.array(taps_all))
    eq_mode_sig = rec_signal.recreate_from_np_array(eq_mode_sig)
    symbs, trace = pilotbased_receiver.pilot_based_cpe(eq_mode_sig[:, pilot_seq_len:frame_length],
                                                           pilot_symbs[:, pilot_seq_len:], pilot_ins_ratio,
                                                              use_pilot_ratio=use_cpe_pilot_ratio, num_average=cpe_average,
                                                       remove_phase_pilots=True)
    #symbs = eq_mode_sig
    #trace = pilotbased_receiver.pilot_based_cpe2(eq_mode_sig, eq_mode_sig.pilots,
                                                       #use_pilot_ratio=use_cpe_pilot_ratio, num_average=cpe_average)


    return np.array(symbs), trace, eq_mode_sig1, taps_all


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
def sim_pilot_txrx(sig_snr, Ntaps=45, beta=0.1, M=256, freq_off = None,cpe_avg=3,
                   frame_length = 2**16, pilot_seq_len = 512, pilot_ins_rat=32,
                   num_frames=3,modal_delay=None, laser_lw = None, fb=20e9,
                   resBits_tx=None, resBits_rx=None):
    
    npols=2

    signal = signals.SignalWithPilots(M, frame_length, pilot_seq_len, pilot_ins_rat, nframes=num_frames, nmodes=npols, fb=fb)

    signal2 = signal.resample(signal.fb*2, beta=beta, renormalise=True)
    # Simulate transmission
    #sig_tx = pilotbased_transmitter.sim_tx(signal2, 2, snr=sig_snr, modal_delay=[800, 200], freqoff=freq_off,
                                                    #resBits_rx=resBits_rx)
    #sig_tx = impairments.simulate_transmission(signal2, snr=sig_snr, modal_delay=[5000,3000], lwdth=laser_lw, freq_off=freq_off, dgd=70e-12, theta=np.pi/6)
    sig_tx = impairments.simulate_transmission(signal2, snr=sig_snr, modal_delay=[5000,3000], lwdth=laser_lw, freq_off=freq_off, dgd=70e-12, theta=np.pi/6)
    #sig_tx = signal2.recreate_from_np_array(sig_tx)



    # Run DSP
    dsp_out, phase, dsp_out2, tall1 = run_pilot_receiver2(sig_tx, os=int(sig_tx.fs / sig_tx.fb), M=M, Numtaps=(17, Ntaps),
                                                  frame_length=sig_tx.frame_len, method=("cma", "sbd"),
                                                  pilot_seq_len=sig_tx.pilot_seq.shape[-1],
                                                  pilot_ins_ratio=sig_tx._pilot_ins_rat, mu=(1e-3, 1e-3),
                                                  cpe_average=cpe_avg, foe_comp=True)
    dsp_out0, phase0, dsp_out20, tall2 = run_pilot_receiver(sig_tx, os=int(sig_tx.fs / sig_tx.fb), M=M, Numtaps=(17, Ntaps),
                                                  frame_length=sig_tx.frame_len, method=("cma", "sbd"),
                                                  pilot_seq_len=sig_tx.pilot_seq.shape[-1],
                                                  pilot_ins_ratio=sig_tx._pilot_ins_rat, mu=(1e-3, 1e-3),
                                                  cpe_average=cpe_avg, foe_comp=True)

    # Calculate GMI and BER
    #ber_res = np.zeros(npols)
    #sout = signal.recreate_from_np_array(np.array(dsp_out[1]))
    #for l in range(npols):
        #gmi_res[l] = signal.cal_gmi(dsp_out[1][:])[0][0]
        #ber_res[l] = signal.cal_ber(np.vstackdsp_out[1][l])
    #gmi_res = sout.cal_gmi()[0]
    #ber_res = sout.cal_ber()
    #dsp_out0 = dsp_out

        
    return dsp_out, dsp_out0, sig_tx, phase, phase0,signal, dsp_out2, dsp_out20, tall1, tall2
    #return dsp_out, signal

if __name__ == "__main__":
    #gmi, ber = sim_pilo    print(methods)t_txrx(20)
    dsp, dsp1, sig, ph, ph2, sign, sig2, sig3, t1, t2= sim_pilot_txrx(30, laser_lw=100e3, freq_off=50e6)
    sigo = dsp
    #sigo = sign.symbols.recreate_from_np_array(dsp)
    sigo1 = sign.recreate_from_np_array(dsp1)
    sigo = helpers.normalise_and_center(sigo)
    sigo1 = helpers.normalise_and_center(sigo1)
    print(sigo.cal_gmi())
    print(sigo1.cal_gmi(signal_rx=sigo1))
