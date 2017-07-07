import numpy as np
import matplotlib.pyplot as plt
from dsp import theory, ber_functions, modulation, utils, equalisation, IO
from scipy.signal import fftconvolve
import sys
import os as os_mod



"""
Test saving and recalling data to hdf files using pytables

"""

hdfn = "/tmp/testsave_load.h5"
snr = np.linspace(5, 30, 4)
wls = [1550, 1552, 1554]
N = 2**16
M = 16
fb = 10
os = 2
fs = os*fb
mu = 1e-3
astep = True
method=("mcma","")
ntaps = 13
beta = 0.1
erows = len(wls)+len(snr)
if os_mod.path.exists(hdfn):
    os_mod.remove(hdfn)
h5f = IO.create_h5_meas_file(hdfn, "w", expectedrows=erows)
#h5f = IO.tb.open_file("sav_recall.h5", "w", "multi wl and OSN msr")
dsp_dict = {"stepsize":(mu, 0), "trsyms":(None, None), "iterations":(1,1), "ntaps":ntaps, "method": ", ".join(method)}
#IO.create_meas_group(h5f, "dual polarisation generated signal", expectedrows=erows)
#IO.create_input_group(h5f, "input symbols", rolloff_dflt=beta, expectedrows=erows)
#IO.create_parameter_group(h5f, "Measurement parameters", expectedrows=erows)

for wl in wls:
    for sr in snr:
        modulator = modulation.QAMModulator(M)
        signalX, symsX, bitsX = modulator.generateSignal(N, sr, samplingrate=fs, baudrate=fb, beta=beta)
        signalY, symsY, bitsY = modulator.generateSignal(N, sr, samplingrate=fs, baudrate=fb, beta=beta)
        signal = np.vstack([signalX, signalY])
        syms = np.vstack([symsX, symsY])
        bits = np.vstack([bitsX, bitsY])
        id = IO.save_osc_meas(h5f, signal, osnr=sr, wl=wl, samplingrate=fs, symbolrate=fb, MQAM=M)
        IO.save_inputs(h5f, id, symbols=syms, bits=bits)
h5f.close()

hf = IO.tb.open_file(hdfn, "a")
hf = IO.create_recvd_data_group(hf, oversampling_dflt=os)
meas_table = hf.root.measurements.oscilloscope.signal
inp_table = hf.root.input.signal
ids = meas_table.cols.id[:]
m_arrays = IO.get_from_table(meas_table, ids, "data")
syms = list(IO.get_from_table(inp_table, ids, "symbols"))
bits = list(IO.get_from_table(inp_table, ids, "bits"))
i = 0
for d_array in m_arrays:
    wx, er =  equalisation.equalise_signal(d_array, os, mu, M, Ntaps=ntaps, adaptive_step=astep, method=method[0])
    signalafter = equalisation.apply_filter(d_array, os, wx )
    evm_x = modulator.cal_EVM(signalafter[0], syms[ids[i]][0])
    evm_y = modulator.cal_EVM(signalafter[1], syms[ids[i]][1])
    ser_x,tmp, data_demod_x = modulator.calculate_SER(signalafter[0], symbol_tx=syms[ids[i]][0])
    ser_y,tmp, data_demod_y = modulator.calculate_SER(signalafter[1], symbol_tx=syms[ids[i]][1])
    ber_x = modulator.cal_BER(signalafter[0], bits[ids[i]][0])[0]
    ber_y = modulator.cal_BER(signalafter[1], bits[ids[i]][1])[0]
    IO.save_recvd(hf, signalafter, ids[i], wx, symbols=np.vstack([data_demod_x, data_demod_y]), evm=(evm_x, evm_y), ber=(ber_x, ber_y), ser=(ser_x, ser_y), dsp_params=dsp_dict)
    i += 1
hf.close()

hf = IO.tb.open_file(hdfn, "r")
pms = hf.root.parameters.experiment.read_where("wl == 1550")
osnr = pms['osnr']
ber = IO.query_table_for_references(hf.root.parameters.experiment, hf.root.analysis.dsp.signal, 'ber', "wl==1550")
snrf = np.linspace(osnr[0], osnr[-1], 100)
plt.plot(osnr, ber, 'o')
plt.plot(snrf, theory.MQAM_BERvsEsN0(10**(snrf/10), M))
plt.show()
hf.close()

