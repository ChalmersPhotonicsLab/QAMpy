from __future__ import division, print_function
import numpy as np
import tables as tb


class tVLArray(tb.VLArray):
    """A variable length array that saves and returns the transpose of the arrays passed to it"""
    _c_classid = "TVLARRAY"
    def __getitem__(self, key):
        a = super(tVLArray, self).__getitem__(key)
        if a.dtype is np.dtype("object"):
            aout = []
            for i in range(len(a)):
                aout.append(np.transpose(aout[i]))
            return np.array(aout)
        else:
            return np.transpose(a)

    def __setitem__(self, key, array):
        if isinstance(key, slice):
            a = []
            for i in range(len(array)):
                a.append(np.transpose(array[i]))
            super(tVLarray, self).__setitem__(key, np.array(a))
        else:
            a = np.transpose(array)
            super(tVLArray, self).__setitem__(key, a)

    def append(self, arr):
        a = np.transpose(arr)
        super(tVLarray, self).append(a)

def create_tvlarray(self, where, name, atom=None, title="", filters=None, expectedrows=None,
                    chunkshape=None, byteorder=None, createparents=False, obj=None):
    """Function to create tvlarrays"""
    pnode = self._get_or_create_path(where, createparents)
    tb.file._checkfilters(filters)
    return tVLArray(pnode, name, atom, title=title, filters=filters, expectedrows=expectedrows,
                chunkshape=chunkshape, byteorder=byteorder)

# register creation of tvlarrays
tb.File.create_vlarray = create_tvlarray

class MDVLarray(tb.VLArray):
    _c_classid = "MDVLARRAY"
    def __getitem__(self, key):
        parent = self._g_getparent()
        shape_array = parent.shapes[key]
        array = super(MDVLarray, self).__getitem__(key)
        if array.dtype is np.dtype("object"):
            a = []
            for i in range(len(shape_array)):
                a.append(array[i].reshape(shape_array[i]))
            return np.array(a)
        else:
            return array.reshape(shape_array)

    def __setitem__(self, key, array):
        parent = self._g_getparent()
        shape_array = parent.shapes
        if isinstance(key, slice):
            a = []
            s = []
            for i in range(len(array)):
                s.append(array[i].shape)
                a.append(array[i].flatten())
            super(MDVLarray, self).__setitem__(key, np.array(a))
            shape_array[key] = np.array(s)
        else:
            shape_array[key] = array.shape
            super(MDVLarray, self).__setitem__(key, array.flatten())

    def append(self, array):
        parent = self._g_getparent()
        shape_array = parent.shapes
        shape_array.append(array.shape)
        super(MDVLarray, self).append(array.flatten())

def create_mdvlarray(self, where, name, atom=None, title="", filters=None, expectedrows=None,
                    chunkshape=None, byteorder=None, createparents=False, obj=None):
    """Function to create a multi dimensional VLArray"""
    pnode = self._get_or_create_path(where, createparents)
    tb.file._checkfilters(filters)
    sharray = tb.VLArray(pnode, "shapes", tb.Int64Atom(), expectedrows=expectedrows)
    return MDVLarray(pnode, name, atom, title=title, filters=filters, expectedrows=expectedrows,
                chunkshape=chunkshape, byteorder=byteorder)

# register creation of mdvlarray
tb.File.create_mdvlarray = create_mdvlarray


def create_parameter_group(h5f, title, description=None, **attrs):
    """
    Create the table for saving measurement parameters

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string
        The title description of the group
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    **attrs:
        other attributes for the table

    Returns
    -------
    h5f : h5filehandle
        Pytables handle to the hdf file
    """
    try:
        gr = h5f.create_group("/", "parameters", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "a")
        gr = h5f.create_group("/", "parameters", title=title)
    if description is None:
        description = {"id":tb.Int64Col(), "osnr":tb.Float64Col(dflt=np.nan),
                  "wl":tb.Float64Col(dflt=np.nan), "symbolrate":tb.Float64Col(),
                  "MQAM": tb.Int64Col(), "Psig": tb.Float64Col(dflt=np.nan)}
    t_param = h5f.create_table(gr, "experiment", description , "measurement parameters")
    if description is None:
        t_param.attrs.symbolrate_unit = "Gbaud"
        t_param.attrs.osnr_unit = "dB"
        t_param.attrs.wl_unit = "nm"
        t_param.attrs.Psig_unit = "dBm"
    for k, v in attrs.items():
        setattr(t_param.attrs, k, v)
    return h5f


def create_meas_group(h5f, title,  nmodes , description=None, **attrs):
    """
    Create the table for saving oscilloscope measurements

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string
        The title description of the group
    data_shape: int
        Number of modes/polarizations
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    **attrs:
        other attributes for the table

    Returns
    -------
    h5f : h5filehandle
        Pytables handle to the hdf file
    """
    try:
        gr_meas = h5f.create_group("/", "measurements", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "a")
        gr_meas = h5f.create_group("/", "measurements", title=title)
    gr_osc = h5f.create_group(gr_osc, "oscilloscope", title=title)
    if description is None:
        description = { "id":tb.Int64Col(), "samplingrate": tb.Float64Col(), "data_shape":tb.Int64Col(shape=(2,)), "data_id": tb.Int64Col()}
    t_meas = h5f.create_table(gr_osc, "signal", description, "sampled signal")
    arr = h5f.create_vlarray(gr_osc, "data", tb.ComplexCol(itemsize=16, shape=(nmodes,)))
    t_meas.attrs.samplingrate_unit = "GS/s"
    for k, v in attrs.items():
        setattr(t_meas.attrs, k, v)
    return h5f

def create_input_group(h5f, title, syms_shape, bits_shape, description=None, **attrs):
    """
    Create the table for saving the input symbols and bits

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string
        The title description of the group
    syms_shape: tuple or None
        Shape of symbols arrays to save (can be None, which means input symbols will not be saved,
        either bits_shape or syms_shape as to be not None)
    bits_shape: tuple or None
        Shape of bits arrays to save (can be None, which means input bits will not be saved,
        either bits_shape or syms_shape as to be not None)
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    **attrs:
        other attributes for the table

    Returns
    -------
    h5f : h5filehandle
        Pytables handle to the hdf file
    """
    assert (syms_shape is not None) or (bits_shape is not None), "Either syms_shape or bits_shape need to be given"
    try:
        gr = h5f.create_group("/", "inputs", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "r+")
        gr = h5f.create_group("/", "inputs", title=title)
    # if no shape for input syms or bits is given use scalar
    if description is None:
        if syms_shape is None and bits_shape is not None:
            description = {"id":tb.Int64Col(), "bits":tb.BoolCol(shape=bits_shape)}
        elif syms_shape is not None and bits_shape is None:
            description = {"id":tb.Int64Col(), "symbols":tb.ComplexCol(itemsize=16, shape=syms_shape)}
        else:
            description = {"id":tb.Int64Col(), "symbols":tb.ComplexCol(itemsize=16, shape=syms_shape),
                  "bits":tb.BoolCol(shape=bits_shape)}
    t_inp = h5f.create_table(gr, "input", description, "input symbols, bits at transmitter")
    for k, v in attrs:
        setattr(t_inp.attrs, k, v)
    return h5f

def create_recvd_data_group(h5f, title, data_shape, description=None, maxtaps=1000, **attrs):
    """
    Create the table for saving recovered data and parameters after DSP

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string
        The title description of the group
    data_shape: tuple
        Shape of data arrays to save
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    maxtaps: int, optional
        maximum number of taps the dsp uses 
    **attrs:
        other attributes for the table

    Returns
    -------
    h5f : h5filehandle
        Pytables handle to the hdf file
    """
    try:
        gr = h5f.create_group("/", "analysis", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "r+")
        gr = h5f.create_group("/", "analysis", title=title)
    if len(data_shape) > 1:
        pols = data_shape[0]
    if description is None:
        dsp_params = { "freq_offset": tb.Float64Col(dflt=np.nan),
                       "freq_offset_N": tb.Int64Col(dflt=np.nan), "phase_est": tb.StringCol(shape=20),
                       "N_angles": tb.Float64Col(dflt=np.nan), "ph_est_blocklength": tb.Int64Col(),
                       "stepsize": tb.Float64Col(shape=2), "trsyms": tb.Float64Col(shape=2),
                       "iterations": tb.Int64Col(shape=2), "taps":tb.ComplexCol(itemsize=64, shape=(pols, maxtaps)),
                       "ntaps": tb.Int64Col()}
        description = {"id":tb.IntCol64(), "data": tb.ComplexCol(itemsize=16, shape=data_shape),
                "symbols":tb.ComplexCol(itemize=16, shape=data_shape),
                "evm": tb.Float64Col(dflt=np.nan), "ber":tb.Float64Col(dflt=np.nan),
                       "ser":tb.Float64Col(dflt=np.nan), "oversampling":tb.Int64Col(),
                       "method": tb.StringCol(shape=20)}
        description.update(dsp_params)
    t_rec = h5f.create_table(gr, "recovered", description, "signal after DSP")
    for k, v in attrs.items():
        setattr(t_rec.attrs, k, v)
    if description is None:
        t_rec.attrs.ber_unit = "dB"
        t_rec.attrs.ser_unit = "dB"
        t_rec.attrs.evm_unit = "%"
    return h5f

def save_inputs(h5file, id_meas, symbols=None, bits=None):
    """
    Save input symbols from the transmitter

    Parameters
    ----------

    h5file : h5filehandle
        pytables file handle
    id_meas: int
        Unique measurement ID to which these inputs belong to.
    symbols: array_like, optional
        Transmitter input symbol array (default=None, note if None bits need to be given)
    bits: array_like, optional
        Transmitter input bit array (default=None, note if None symbols need to be given)

    """
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
    """
    Save measured data from oscilloscope

    Parameters
    ----------

    h5file : h5filehandle
        pytables file handle
    data: array_like
        The sampled signal array, needs to be the same shape as defined when creating the group
    id_meas: int
        Unique measurement ID
    osnr: Float, optional
        Optical Signal to Noise Ratio of the measurement in dB
    wl: Float, optional
        Wavelength of the measurement in nm
    Psig: Float, optional
        Signal power at the receiver
    samplingrate: Float, optional
        Sampling rate of the signal in GS/s
    symbolrate: Float, optional
        Symbolrate of the signal in Gbaud
    MQAM: Int, optional
        QAM order of the signal
    """
    meas_table = h5file.root.measurements.oscilloscope
    m_row = meas_table.row
    m_row['id'] = id_meas
    m_row['data'] = data
    m_row['samplingrate'] = samplingrate
    m_row.append()
    meas_table.flush()
    par_table = h5file.root.parameters.experiment
    par_cols = {"osnr": osnr, "wl": wl, "Psig": Psig, "symbolrate": symbolrate, "MQAM": MQAM}
    p_row = par_table.row
    p_row['id'] = id_meas
    for k, v in par_cols.items():
        if v is not None:
            p_row[k] = v
    p_row.append()
    par_table.flush()

def save_recvd(h5file, data, id_meas, symbols=None, oversampling=None, evm=None, ber=None, ser=None, dsp_params=None):
    """
    Save recovered data after DSP

    Parameters
    ----------

    h5file : h5filehandle
        pytables file handle
    data: array_like
        The sampled signal array, needs to be the same shape as defined when creating the group
    id_meas: int
        Unique measurement ID this corresponds to
    symbols: array_like, optional
        Recovered symbols
    oversampling: Int, optional
        Oversampling used for recovery
    evm: Float, optional
        Error Vector Magnitude of the signal in percent
    ber: Float, optional
        Bit Error Rate of the signal in percent
    ser: Float, optional
        Symbol Error Rate of the signal in percent
    dsp_params: dict
        DSP parameters used for recovery. See the description in the recovery group creation for the definition.
    """
    rec_table = h5file.root.recovered.analysis
    cols = {"evm": evm, "ber": ber, "ser": ser}
    row = rec_table.row
    row['id'] = id_meas
    row['data'] = data
    row['oversampling'] = oversampling
    for k, v in cols.items():
        if v is not None:
            row[k] = v
    if dsp_params is not None:
        for k, v in dsp_params:
            if k == "taps":
                vv = np.atleast_2d(v)
                row[k][:, :vv.shape[1]] = vv
            row[k] = v
    row.append()
    rec_table.flush()
