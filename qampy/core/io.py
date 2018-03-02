from __future__ import division, print_function
import numpy as np
import tables as tb

PARAM_UNITS = {"symbolrate_unit": "Gbaud", "osnr_unit": "dB", "wl_unit": "nm", "Psig_unit": "dBm"}
MEAS_UNITS = {"samplingrate_unit": "GS/s"}
DSP_UNITS = {"ber_unit": "dB", "ser_unit": "dB", "evm_unit": "%"}

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
    """
    A multi dimensional variable length array. The shape of the array is saved as "name"_shape
    in the same group as this array. To create it automatically use the create_mdvlarray function
    of the h5 file.
    """
    _c_classid = "MDVLARRAY"
    def __getitem__(self, key):
        parent = self._g_getparent()
        shape_array = getattr(parent, self.name+"_shape")[key]
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
        shape_array = getattr(parent, name+"_shape")
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
        shape_array = getattr(parent, self.name+"_shape")
        shape_array.append(array.shape)
        super(MDVLarray, self).append(array.flatten())

def create_mdvlarray(self, where, name, atom=None, title="", filters=None, expectedrows=None,
                    chunkshape=None, byteorder=None, createparents=False, obj=None):
    """Function to create a multi dimensional VLArray"""
    pnode = self._get_or_create_path(where, createparents)
    tb.file._checkfilters(filters)
    sharray = tb.VLArray(pnode, name+"_shape", tb.Int64Atom(), expectedrows=expectedrows)
    return MDVLarray(pnode, name, atom, title=title, filters=filters, expectedrows=expectedrows,
                chunkshape=chunkshape, byteorder=byteorder)

# register creation of mdvlarray
tb.File.create_mdvlarray = create_mdvlarray

def create_h5_meas_file(fn, title, filters=tb.Filters(complevel=9, complib="blosc:lz4", fletcher32=True), create_rec_group=False, **kwargs):
    """
    Create a h5 file for saving measurement data.

    Parameters
    ----------

    fn : string
        file name
    title: string
        description title for the file
    filters: tables.Filters instance, optional
        compression filters used by h5f2
    create_rec_group: bool, optional
        whether to create the recovered data group, default is not to do this
    **kwargs
        kword arguments to be passed to the group creation (best to set expectedrows to a sensible value)

    Returns
    -------
    h5f: tables file instance
        the hdf file instance for pytables 
    """
    h5f = tb.open_file(fn, "w", title, filters=filters)
    h5f = create_parameter_group(h5f, **kwargs)
    h5f = create_meas_group(h5f, **kwargs)
    h5f = create_input_group(h5f, **kwargs)
    if create_rec_group:
        h5f = create_recvd_data_group(h5f, **kwargs)
    return h5f

def create_parameter_group(h5f, title="parameters of the measurement", description=None, attrs=PARAM_UNITS, **kwargs):
    """
    Create the table for saving measurement parameters

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string, optional
        The title description of the group
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    attrs: dict, optional
        dictionary of attributes to set on the table
    **kwargs:
        keyword arguments passed to create_table, it is highly recommended to set expectedrows

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
                       "MQAM": tb.Int64Col(), "Psig": tb.Float64Col(dflt=np.nan), "L":tb.Float64Col(dflt=0)}
    t_param = h5f.create_table(gr, "experiment", description , "measurement parameters", **kwargs)
    for k, v in attrs.items():
        setattr(t_param.attrs, k, v)
    return h5f


def create_meas_group(h5f, title="measurement data",  description=None, attrs=MEAS_UNITS, arrays=["data"], **kwargs):
    """
    Create the table for saving oscilloscope measurements

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string, optional
        The title description of the group
    data_shape: int
        Number of modes/polarizations
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    arrays: list, optional
        name of arrays referenced in the table
    attrs: dict, optional
        attributes on the table
    **kwargs:
        keyword arguments passed to create_table/array, it is highly recommended to set expectedrows

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
    gr_osc = h5f.create_group(gr_meas, "oscilloscope", title="Data from Realtime oscilloscope")
    if description is None:
        description = { "id":tb.Int64Col(), "samplingrate": tb.Float64Col(), "idx_data": tb.Int64Col()}
    t_meas = h5f.create_table(gr_osc, "signal", description, "sampled signal", **kwargs)
    setattr(t_meas.attrs, "arrays", arrays)
    arr = h5f.create_mdvlarray(gr_osc, "data", tb.ComplexAtom(itemsize=16), **kwargs)
    for k, v in attrs.items():
        setattr(t_meas.attrs, k, v)
    return h5f

def create_input_group(h5f, title="input data at transmitter", rolloff_dflt=np.nan, attrs={}, arrays=["symbols", "bits"], **kwargs):
    """
    Create the table for saving the input symbols and bits

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string, optional
        The title description of the group
    attrs: dict, optional
        attributes on the table
    arrays: list, optional
        name of arrays referenced in the table
    **kwargs:
        keyword arguments passed to create_table/array, it is highly recommended to set expectedrows

    Returns
    -------
    h5f : h5filehandle
        Pytables handle to the hdf file
    """
    try:
        gr = h5f.create_group("/", "input", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "a")
        gr = h5f.create_group("/", "input", title=title)
    # if no shape for input syms or bits is given use scalar
    t_in = h5f.create_table(gr, "signal", {"id": tb.Int64Col(), "idx_symbols": tb.Int64Col(dflt=0),
                                               "idx_bits": tb.Int64Col(dflt=0), "rolloff": tb.Float64Col(dflt=rolloff_dflt)}, title="parameters of input signal", **kwargs)
    setattr(t_in.attrs, "arrays", arrays)
    arr_syms = h5f.create_mdvlarray(gr, "symbols", tb.ComplexAtom(itemsize=16, dflt=np.nan), title="sent symbols", **kwargs)
    arr_bits = h5f.create_mdvlarray(gr, "bits", tb.BoolAtom(), title="sent bits", **kwargs)
    for k, v in attrs:
        setattr(t_in.attrs, k, v)
    return h5f

def create_recvd_data_group(h5f, title="data analysis and qampy results", description=None, oversampling_dflt=2,
                            attrs=DSP_UNITS, arrays=["data", "symbols", "taps", "bits"], nmodes=2, **kwargs):
    """
    Create the table for saving recovered data and parameters after DSP

    Parameters
    ----------

    h5f : string or h5filehandle
        The file to use, if a string create or open new file
    title: string
        The title description of the group
    description: dict or tables.IsDescription (optional)
        If given use to create the table
    attrs: dict, optional
        attributes for the table
    arrays: list, optional
        name of arrays referenced in the table
    nmodes: int, optional
        number of modes/polarisations
    **kwargs:
        keyword arguments passed to create_table/array, it is highly recommended to set expectedrows

    Returns
    -------
    h5f : h5filehandle
        Pytables handle to the hdf file
    """
    try:
        gr = h5f.create_group("/", "analysis", title=title)
    except AttributeError:
        h5f = tb.open_file(h5f, "a")
        gr = h5f.create_group("/", "analysis", title=title)
    gr_dsp = h5f.create_group(gr, "qampy", title="Signal from DSP")
    if description is None:
        dsp_params = { "freq_offset": tb.Float64Col(dflt=np.nan),
                       "freq_offset_N": tb.Int64Col(dflt=0), "phase_est": tb.StringCol(itemsize=20),
                       "N_angles": tb.Float64Col(dflt=np.nan), "ph_est_blocklength": tb.Int64Col(),
                       "stepsize": tb.Float64Col(shape=2), "trsyms": tb.Float64Col(shape=2),
                       "iterations": tb.Int64Col(shape=2),
                       "ntaps": tb.Int64Col(),
                       "method": tb.StringCol(itemsize=20)}
        description = {"id":tb.Int64Col(), "idx_data": tb.Int64Col(), "idx_symbols": tb.Int64Col(),
                       "idx_bits": tb.Int64Col(), "idx_taps": tb.Int64Col(),
                       "evm": tb.Float64Col(dflt=np.nan, shape=nmodes), "ber":tb.Float64Col(dflt=np.nan, shape=nmodes),
                       "ser":tb.Float64Col(dflt=np.nan, shape=nmodes), "oversampling":tb.Int64Col(dflt=oversampling_dflt)}
        description.update(dsp_params)
    t_rec = h5f.create_table(gr_dsp, "signal", description, "signal after DSP", **kwargs)
    setattr(t_rec.attrs, "arrays", arrays)
    data_arr = h5f.create_mdvlarray(gr_dsp, "data", tb.ComplexAtom(itemsize=16), "signal after DSP", **kwargs)
    syms_arr = h5f.create_mdvlarray(gr_dsp, "symbols", tb.ComplexAtom(itemsize=16, dflt=np.nan), "recovered symbols", **kwargs)
    taps_arr = h5f.create_mdvlarray(gr_dsp, "taps", tb.ComplexAtom(itemsize=16, dflt=np.nan), "qampy taps", **kwargs)
    bits_arr = h5f.create_mdvlarray(gr_dsp, "bits", tb.BoolAtom(dflt=False), "recovered bits", **kwargs)
    for k, v in attrs.items():
        setattr(t_rec.attrs, k, v)
    return h5f

def save_array_to_table(table, name, array):
    parent = table._g_getparent()
    data_stor = getattr(parent, name)
    array = np.asarray(array)
    data_stor.append(array)
    return data_stor.nrows - 1 # index from 0

def save_inputs(h5file, id_meas, symbols=None, bits=None, rolloff=None):
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
    input_tb = h5file.root.input.signal
    row = input_tb.row
    row['id'] = id_meas
    if symbols is not None:
        row['idx_symbols'] = save_array_to_table(input_tb, "symbols", symbols)
    if bits is not None:
        row['idx_bits'] = save_array_to_table(input_tb, "bits", bits)
    row['rolloff'] = rolloff
    row.append()
    input_tb.flush()

def save_osc_meas(h5file, data,  osnr=None, wl=None, measurementN=0, Psig=None, samplingrate=None, symbolrate=None, MQAM=None):
    """
    Save measured data from oscilloscope

    Parameters
    ----------

    h5file : h5filehandle
        pytables file handle
    data: array_like
        The sampled signal array, needs to be the same shape as defined when creating the group
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

    Returns
    -------

    id_meas: int
        Unique measurement ID
    """
    meas_table = h5file.root.measurements.oscilloscope.signal
    m_row = meas_table.row
    id_meas = save_array_to_table(meas_table, "data", data)
    m_row['id'] = id_meas
    m_row['idx_data'] = id_meas
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
    return id_meas

def save_recvd(h5file, data, id_meas, taps, symbols=None, bits=None, oversampling=None, evm=None, ber=None, ser=None, dsp_params=None):
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
    taps: array_like
        The equaliser taps
    symbols: array_like, optional
        Recovered symbols
    bits: array_like, optional
        Recovered bits
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
    rec_table = h5file.root.analysis.dsp.signal
    cols = {"evm": evm, "ber": ber, "ser": ser}
    row = rec_table.row
    row['id'] = id_meas
    if oversampling is not None:
        row['oversampling'] = oversampling
    row['idx_data'] = save_array_to_table(rec_table, "data", data)
    row['idx_taps'] = save_array_to_table(rec_table, "taps", taps)
    if symbols is not None:
        row['idx_symbols'] = save_array_to_table(rec_table, "symbols", symbols)
    if bits is not None:
        row['idx_bits'] = save_array_to_table(rec_table, "bits", bits)
    for k, v in cols.items():
        if v is not None:
            row[k] = v
    if dsp_params is not None:
        for k, v in dsp_params.items():
            row[k] = v
    row.append()
    rec_table.flush()

def _array_idx_iterator(array, idxs):
    i = 0
    while i < len(idxs):
        yield array[idxs[i]]
        i += 1

def array_from_table(table, name, query=None):
    """
    Get arrays from table

    Parameters
    ----------
    table : pytables table
        table to get this from
    name : string
        name of the array to get
    query: string, optional
        query string for selecting specific array rows (default=None select all rows)

    Returns
    -------
    arrays: iterator
       iterator over the arrays
    """
    pnode = table._g_getparent()
    arr_stor = getattr(pnode, name)
    colname = "idx_"+name
    if query is None:
        idx = getattr(table.cols, colname)[:]
    else:
        idx = [r[colname] for r in table.where(query)]
    return _array_idx_iterator(arr_stor, idx)

def construct_id_query(ids, name="id"):
    """
    Construct a query for a table based on a list of given ids (e.g. from a different table)

    Parameters
    ----------
    ids : list
        list of integer ids to use
    name: string, optional
        column name of the ids to query

    Returns
    -------
    query: string
        query string that can be used with table.where
    """
    return "|".join(["({0}=={1})".format(name,id) for id in ids])

def get_from_table(table, ids, name, id_col="id"):
    """
    Get values for a given list of ids from table

    Parameters
    ----------
    table: pytables table
        table containing the array reference
    ids: list
        list of array reference ids
    name: string
        name of desired values column
    id_col: string, optional
        column containing the array references

    Returns
    -------
    result: iterator or list
       values at the desired ids. If results are an array this is an iterator otherwise it is a list
    """
    if name in table.attrs.arrays:
        return array_from_table(table, name, query=construct_id_query(ids, name=id_col))
    else:
        return [x[name] for x in table.where(construct_id_query(ids))]

def query_table_for_references(table_query, table_res, colname, query, id_col="id"):
    """
    Query a table for references and get result from second table based on query on another table, for example query theoretical
    parameter table for a given OSNR range and get the corresponding 

    Parameters
    ----------
    table_query: pytables table
        table to query for references
    table_res: pytables table
        table to get the result from
    colname: string
        column name of the result
    query: string
        query to run on table_query
    id_col: string, optional
        name of the reference id column

    Returns
    -------
    results: iterator or list
        query results. If the results are arrays this is an iterator, otherwise a list
    """

    ids = table_query.read_where(query)[id_col]
    return get_from_table(table_res, ids, colname, id_col=id_col)
