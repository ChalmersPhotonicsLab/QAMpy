import numpy as np
from dsp import equalisation, modulation, utils, phaserecovery, pilotbased_receiver
import matplotlib.pylab as plt

def run_pilot_receiver(rec_signal, pilot_symbs, process_frame_id=0, sh=False, do_extra_blind_eq=False,
                       do_extra_blind_BPS=(False, 64, 128), os=2, M=128, Numtaps=(17, 45), frame_length=2 ** 16,
                       method=('cma', 'cma'), pilot_seq_len=512, pilot_ins_ratio=32, Niter=(10, 30), mu=(1e-3, 1e-3),
                       adap_step=(True, True), cpe_average=5, use_cpe_pilot_ratio=1, remove_inital_cpe_output=True,
                       remove_phase_pilots=True, do_pilot_based_foe=True, foe_symbs=None, blind_foe_payload=False):
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

    # Extra CPE Using blind phase search
    if do_extra_blind_BPS[0]:
        phase_comp_symbs_bps = []
        phase_trace_bps = []
        QAM = modulation.QAMModulator(M)
        for l in range(npols):
            symbs, phase = phaserecovery.blindphasesearch(dsp_sig_out[l].flatten(), do_extra_blind_BPS[1], QAM.symbols,
                                                          do_extra_blind_BPS[2])
            phase_comp_symbs_bps.append(symbs)
            phase_trace_bps.append(phase)
        dsp_sig_out = phase_comp_symbs_bps

    # Extra final blind equalizer
    if do_extra_blind_eq:
        symbs_out = []
        for l in range(npols):
            wx, err_both = equalisation.equalise_signal(dsp_sig_out[l], 1, 1e-3, M, Niter=10, Ntaps=Numtaps[1],
                                                        method="mcma", adaptive_stepsize=True)
            symbs_out.append(utils.orthonormalize_signal(equalisation.apply_filter(dsp_sig_out[l], 1, wx)))
        dsp_sig_out = symbs_out

    return eq_pilots, dsp_sig_out, shift_factor, taps, phase_trace, foePerMode
