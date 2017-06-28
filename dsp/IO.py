from __future__ import division, print_function
import numpy as np
import tables as tb


def osc_data_factory(shape):
    class OscilloscopeData(tb.IsDescription):
        id_meas = tb.Int64Col()
        data = tb.ComplexCol(itemsize=16, shape=shape)
        samplingrate = tb.Float64Col()
    return OscilloscopeData

def rec_data_factory(shape):
    class RecoveredData(tb.IsDescription):
        id_meas = tb.Int64Col()
        data = tb.ComplexCol(itemsize=16, shape=shape)
        symbols = tb.ComplexCol(itemsize=16, shape=shape)
        evm = tb.Float64Col(dflt=np.nan)
        ber = tb.Float64Col(dflt=np.nan)
        ser = tb.Float64Col(dflt=np.nan)
        oversampling = tb.Int32Col(dflt=2)
    return Recovereddata

def input_data_factory(shape, shape_bits):
    class InputData(tb.IsDescription):
        id_meas = tb.Int64Col()
        symbols = tb.ComplexCol(itemsize=16, shape=shape)
        bits = tb.BoolCol(shape=shape_bits)
    return InputData

class Parameters(tb.IsDescription):
    id_meas = tb.Int64Col()
    osnr = tb.Float64Col(dflt=np.nan)
    wl = tb.Float64Col(dflt=np.nan)
    symbolrate = tb.Float64Col()
    MQAM = tb.Int64Col()
    measurementN = tb.Int64Col(dflt=0)
    Psig = tb.Float64Col(dflt=np.nan)



def create_meas_file(fn, title, description, data_shape, input_syms_shape=None, input_attrs=None, input_bits_shape=None, **attrs):
    h5f = tb.open_file(fn, 'w', title=title)
    gr_meas = h5f.create_group("/", "measurements", title=description)
    gr_inp = h5f.create_group("/", "inputs", title="input symbols or bits")
    t_meas = h5f.create_table(gr_meas, "oscilloscope", osc_data_factory(data_shape), "sampled signal")
    t_meas.attrs.samplingrate_unit = "GS/s"
    # if no shape for input syms or bits is given use scalar
    input_syms_shape = input_syms_shape or 1
    input_bits_shape = input_bits_shape or 1
    t_inp = h5f.create_table(gr_inp, "input", input_data_factory(input_syms_shape, input_bits_shape), "input symbols, bits at transmitter")
    if input_attrs is not None:
        for k, v in input_attrs:
            setattr(t_inp.attrs, k, v)
    t_param = h5f.create_table("/", "parameters", Parameters, "measurement parameters")
    t_param.attrs.symbolrate_unit = "Gbaud"
    t_param.attrs.osnr_unit = "dB"
    t_param.attrs.wl_unit = "nm"
    t_param.attrs.Psig_unit = "dBm"
    for k, v in attrs.items():
        setattr(t_meas.attrs, k, v)
    return h5f

def create_recvd_data_group(h5f, description, data_shape, **attrs):
    try:
        gr = h5f.create_group("/", "analysis", title=description)
    except AttributeError:
        h5f = tb.open_file(h5f, "r+")
        gr = h5f.create_group("/", "analysis", title=description)
    t_rec = h5f.create_table(gr, "recovered", rec_data_factory(data_shape), "signal after DSP")
    for k, v in attrs.items():
        setattr(t_rec.attrs, k, v)
    t_rec.attrs.ber_unit = "dB"
    t_rec.attrs.ser_unit = "dB"
    t_rec.attrs.evm_unit = "%"
    return h5f

def save_osc_meas(h5file, data, id_meas,  osnr=None, wl=None, measurementN=0, Psig=None, samplingrate=None, symbolrate=None, MQAM=None):
    meas_table = h5file.root.measurements.oscilloscope
    m_row = meas_table.row
    m_row['id_meas'] = id_meas
    m_row['data'] = data
    m_row['samplingrate'] = samplingrate
    m_row.append()
    meas_table.flush()
    par_table = h5file.root.parameters
    par_cols = {"osnr": osnr, "wl": wl, "measurementN": measurementN, "Psig": Psig, "symbolrate": symbolrate, "MQAM": MQAM}
    p_row = par_table.row
    p_row['id_meas'] = id_meas
    for k, v in par_cols.items():
        if v is not None:
            p_row[k] = v
    p_row.append()
    par_table.flush()

def save_recvd(h5file, data, id_meas, oversampling=None, evm=None, ber=None, ser=None):
    rec_table = h5file.root.recovered.analysis
    cols = {"evm": evm, "ber": ber, "ser": ser}
    row = rec_table.row
    row['id_meas'] = id_meas
    row['data'] = data
    row['oversampling'] = oversampling
    for k, v in cols.items():
        if v is not None:
            row[k] = v
    row.append()
    rec_table.flush()

