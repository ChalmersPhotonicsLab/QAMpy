from __future__ import division, print_function
import numpy as np
import tables as tb

def create_parameter_group(h5f, title, description=None, **attrs):
    try:
        gr = h5f.create_group("/", "parameters", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "a")
        gr = h5f.create_group("/", "parameters", title=title)
    if description is None:
        description = {"id":tb.Int64Col(), "osnr":tb.Float64Col(dflt=np.nan),
                  "wl":tb.Float64Col(dflt=np.nan), "symbolrate":tb.Float64Col(),
                  "MQAM": tb.Int64Col(), "measurementN":tb.Int64Col(dflt=0),
                  "Psig": tb.Float64Col(dflt=np.nan)}
    t_param = h5f.create_table(gr, "experiment", description , "measurement parameters")
    if description is None:
        t_param.attrs.symbolrate_unit = "Gbaud"
        t_param.attrs.osnr_unit = "dB"
        t_param.attrs.wl_unit = "nm"
        t_param.attrs.Psig_unit = "dBm"
    for k, v in attrs.items():
        setattr(t_param.attrs, k, v)


def create_meas_group(h5f, title,  data_shape, description=None, **attrs):
    try:
        gr_meas = h5f.create_group("/", "measurements", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "a")
        gr_meas = h5f.create_group("/", "measurements", title=title)
    if description is None:
        description = { "id":tb.Int64Col(), "data": tb.ComplexCol(itemsize=16, shape=data_shape), "samplingrate": tb.Float64Col()}
    t_meas = h5f.create_table(gr_meas, "oscilloscope", description, "sampled signal")
    t_meas.attrs.samplingrate_unit = "GS/s"
    for k, v in attrs.items():
        setattr(t_meas.attrs, k, v)
    return h5f

def create_input_group(h5f, title, syms_shape, bits_shape, description=None, **attrs):
    try:
        gr = h5f.create_group("/", "inputs", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "r+")
        gr = h5f.create_group("/", "inputs", title=title)
    # if no shape for input syms or bits is given use scalar
    syms_shape = syms_shape or 1
    bits_shape = bits_shape or 1
    if description is None:
        description = {"id":tb.Int64Col(), "symbols":tb.ComplexCol(itemsize=16, shape=syms_shape),
                  "bits":tb.BoolCol(shape=bits_shape)}
    t_inp = h5f.create_table(gr, "input", description, "input symbols, bits at transmitter")
    for k, v in attrs:
        setattr(t_inp.attrs, k, v)

def create_recvd_data_group(h5f, title, data_shape, description=None, **attrs):
    try:
        gr = h5f.create_group("/", "analysis", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "r+")
        gr = h5f.create_group("/", "analysis", title=title)
    if description is None:
        description = {"id":tb.IntCol64(), "data": tb.ComplexCol(itemsize=16, shape=data_shape),
                "symbols":tb.ComplexCol(itemize=16, shape=data_shape),
                "evm": tb.Float64Col(dflt=np.nan), "ber":tb.Float64Col(dflt=np.nan),
                "ser":tb.Float64Col(dflt=np.nan), "oversampling":tb.Int64Col()}
    t_rec = h5f.create_table(gr, "recovered", description, "signal after DSP")
    for k, v in attrs.items():
        setattr(t_rec.attrs, k, v)
    if description is None:
        t_rec.attrs.ber_unit = "dB"
        t_rec.attrs.ser_unit = "dB"
        t_rec.attrs.evm_unit = "%"
    return h5f

def save_inputs(h5file, id_meas, symbols=None, bits=None):
    assert (symbols is not None) or (bits is not None), "Either input symbols or bits need to be given"
    input_tb = h5file.root.inputs.input
    row = input_tb.row
    row['id'] = id_meas
    if symbols is not None:
        row['symbols'] = symbols
    if bits is not None:
        row['bits'] = bits
    row.append()
    input_tb.flush()

def save_osc_meas(h5file, data, id_meas,  osnr=None, wl=None, measurementN=0, Psig=None, samplingrate=None, symbolrate=None, MQAM=None):
    meas_table = h5file.root.measurements.oscilloscope
    m_row = meas_table.row
    m_row['id'] = id_meas
    m_row['data'] = data
    m_row['samplingrate'] = samplingrate
    m_row.append()
    meas_table.flush()
    par_table = h5file.root.parameters.experiment
    par_cols = {"osnr": osnr, "wl": wl, "measurementN": measurementN, "Psig": Psig, "symbolrate": symbolrate, "MQAM": MQAM}
    p_row = par_table.row
    p_row['id'] = id_meas
    for k, v in par_cols.items():
        if v is not None:
            p_row[k] = v
    p_row.append()
    par_table.flush()

def save_recvd(h5file, data, id_meas, oversampling=None, evm=None, ber=None, ser=None):
    rec_table = h5file.root.recovered.analysis
    cols = {"evm": evm, "ber": ber, "ser": ser}
    row = rec_table.row
    row['id'] = id_meas
    row['data'] = data
    row['oversampling'] = oversampling
    for k, v in cols.items():
        if v is not None:
            row[k] = v
    row.append()
    rec_table.flush()

