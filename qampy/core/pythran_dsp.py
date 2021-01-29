import numpy as np

def cabsq(x):
    return x.real**2 + x.imag**2


def cal_exp_sum(sym , syms, z, sigma):
    N = syms.size
    out = 0.
    for i in range(N):
        out += np.exp(-sigma*(2*np.real(z*(sym-syms[i])) +
                              abs(sym-syms[i])**2))
    return out

#we need to use this version because of #1133 which made argmin significantly slower
def det_symbol(X, symbs):
    d0 = 1000.
    for j in range(symbs.shape[0]):
        d = cabsq(X-symbs[j])
        if d < d0:
            d0 = d
            s = symbs[j]
    return s, d0


def select_angle_index(x, N):
    L,M = x.shape
    csum = np.zeros((L, M), dtype=x.dtype)
    idx = np.zeros(L, dtype=np.int32)
    for i in range(1, L):
        dmin = 1000.
        if i < N:
            for k in range(M):
                csum[i, k] = csum[i-1, k] + x[i, k]
        else:
            for k in range(M):
                csum[i, k] = csum[i-1, k] + x[i, k]
                dtmp = csum[i, k] - csum[i-N, k]
                if dtmp < dmin:
                    idx[i-N//2] = k
                    dmin = dtmp
    return idx


#pythran export bps(complex128[], float64[][], complex128[], int)
#pythran export bps(complex64[], float32[][], complex64[], int)
def bps(E, testangles, symbols, N):
    """
    Blind phase search algorithm
    
    Parameters
    ----------
    E : array_like
        input signal
    testangles : array_like
        set of test angles
    symbols : array_like
        symbol alphabet
    N : int
        averaging filter

    Returns
    -------
    phase : array_like
        estimated phase vector
    """
    L = E.shape[0]
    p = testangles.shape[0]
    assert p == 0 or p == L, "p must be either 0 or the length of the input signal"
    M = symbols.shape[0]
    Ntestangles = testangles.shape[1]
    comp = np.exp(1j*testangles)
    dists = np.zeros((L, Ntestangles), dtype=testangles.dtype)+100.
    #omp parallel for
    for i in range(L):
        if p > 1:
            ph_idx = i
        else:
            ph_idx = 0
        for j in range(Ntestangles):
            tmp = E[i] * comp[ph_idx, j]
            s, dtmp = det_symbol(tmp, symbols)
            if dtmp < dists[i, j]:
                dists[i, j] = dtmp
    return select_angle_index(dists, 2*N)

def cal_l_values(btx, rx, snr):
    tmp = np.sum(np.exp(-snr*abs(btx - rx)**2), axis=0)
    return np.log(tmp[1]) - np.log(tmp[0])

#pythran export soft_l_value_demapper(complex128[], int, float64, complex128[][][])
#pythran export soft_l_value_demapper(complex128[], int, float32, complex128[][][])
#pythran export soft_l_value_demapper(complex64[], int, float32, complex64[][][])
#pythran export soft_l_value_demapper(complex64[], int, float64, complex64[][][])
def soft_l_value_demapper(rx_symbs, num_bits, snr, bits_map):
    assert bits_map.shape[0] >= num_bits
    N = rx_symbs.shape[0]
    L_values = np.zeros((N, num_bits))
    k = bits_map.shape[1]
    #omp parallel for collapse(2)
    for symb in range(N):
        for bit in range(num_bits):
            L_values[symb, bit] = cal_l_values(bits_map[bit,:,:], rx_symbs[symb], snr)
    return L_values

def find_minmax(btx, rx):
    tmp = 10000.
    tmp2 = 10000.
    k = btx.shape[0]
    for l in range(k):
        tmp3 = abs(btx[l,1] - rx)**2
        tmp4 = abs(btx[l,0] - rx)**2
        if tmp3 < tmp:
            tmp = tmp3
        if tmp4 < tmp2:
           tmp2 = tmp4
    return tmp, tmp2

#pythran export soft_l_value_demapper_minmax(complex128[], int, float64, complex128[][][])
#pythran export soft_l_value_demapper_minmax(complex64[], int, float32, complex64[][][])
def soft_l_value_demapper_minmax(rx_symbs, num_bits, snr, bits_map):
    assert bits_map.shape[0] >= num_bits
    N = rx_symbs.shape[0]
    L_values = np.zeros((N, num_bits))
    k = bits_map.shape[1]
    #omp parallel for collapse(2)
    for symb in range(N):
        for bit in range(num_bits):
            tmp, tmp2 = find_minmax(bits_map[bit,:,:], rx_symbs[symb])
            L_values[symb,bit] = snr*(tmp2-tmp)
    return L_values

#pythran export select_angles(float64[][], int[])
#pythran export select_angles(float32[][], int[])
#pythran export select_angles(float64[][], int32[])
#pythran export select_angles(float32[][], int32[])
def select_angles(angles, idx):
    L = angles.shape[0]
    assert idx.shape[0] >= L # this will be removed upon compilation but can yield a speedup
    assert np.max(idx) < angles.shape[-1] # this will be removed upon compilation but can yield a speedup
    anglesn = np.zeros(L, dtype=angles.dtype)
    if angles.shape[0] > 1:
        #omp parallel for
        for i in range(L):
            anglesn[i] = angles[i, idx[i]]
        return anglesn
    else:
        L = idx.shape[0]
        anglesn = np.zeros(L, dtype=angles.dtype)
        #omp parallel for
        for i in range(L):
            anglesn[i] = angles[0, idx[i]]
    return anglesn

#pythran export prbs_ext(int, int[], int, int)
def prbs_ext(seed, taps, nbits, N):
    out = np.zeros(N, dtype=np.uint8)
    sr = seed
    for i in range(N):
        xor = 0
        for t in taps:
            if (sr & (1 << (nbits-t))) != 0:
                xor ^= 1
        sr = (xor << nbits -1 ) + (sr >> 1)
        out[i] = xor
    return out

#pythran export prbs_int(int, int, int, int)
def prbs_int(seed, mask, nbits, N):
    out = np.zeros(N, dtype=np.uint8)
    state = seed
    for i in range(N):
        state = (state << 1)
        xor = state >> nbits
        if xor != 0:
            state ^= mask
        out[i] = xor
    return out

#pythran export cal_gmi_mc(complex128[], float64, int, complex128[][][])
def cal_gmi_mc(symbols, snr, ns, bit_map):
    M = symbols.size
    nbits = int(np.log2(M))
    assert bit_map.shape[0] >= nbits, "bit map must have entry for each bit"
    assert bit_map.shape[2] == 2, "bit map must have 0 and 1 bit entry"
    gmi = 0
    z = np.sqrt(1/snr)*(np.random.randn(ns) +
                        1j*np.random.randn(ns))/np.sqrt(2)
    #omp parallel for collapse(3) reduction(+:gmi)
    for k in range(nbits):
        for b in range(2):
            for l in range(ns):
                for sym in bit_map[k, :, b]:
                    nom = cal_exp_sum(sym, symbols, z[l], snr)
                    denom = cal_exp_sum(sym, bit_map[k, :, b], z[l], snr)
                    gmi += np.log2(nom/denom)
    return nbits-gmi/(M*ns)

#pythran export cal_lut_avg(complex128[], int[], int[], int)
#pythran export cal_lut_avg(complex64[], int[], int[], int)
def cal_lut_avg(err, idx_I, idx_Q,  N):
    """
    Calculate average pattern lookup tables. This looks at all the patterns for the In-phase and
    Quadrature components of the field and calculates the average error for all patterns. Importantly 
    the indices need to be aligned so that the error and the corresponding pattern have the same index
    
    Parameters
    ----------
    err : array_like
        error per signal symbol 
    idx_I : array_like
        pattern index for the in-phase components
    idx_Q : array_like
        pattern index for the quadrature components
    N : int
        number of unique patterns

    Returns
    -------
        err_avg : array_like
            average error for all patterns. Patterns that do not appear in idx_I or idx_Q will be 0.
    """
    L = err.size
    assert idx_I.shape[0] > L
    assert idx_Q.shape[0] > L
    err_avg_I = np.zeros(N, dtype=err.real.dtype)
    err_avg_Q = np.zeros(N, dtype=err.real.dtype)
    nI = np.zeros(N, dtype=int)
    nQ = np.zeros(N, dtype=int)
    for i in range(L):
        err_avg_I[idx_I[i]] += err[i].real
        err_avg_Q[idx_Q[i]] += err[i].imag
        nI[idx_I[i]] += 1
        nQ[idx_Q[i]] += 1
    for i in range(N):
        if nI[i] == 0:
            nI[i] = 1
        if nQ[i] == 0:
            nQ[i] = 1
    return err_avg_I/nI + 1j* err_avg_Q/nQ

#pythran export estimate_snr(complex128[], complex128[], complex128[])
#pythran export estimate_snr(complex64[], complex64[], complex64[])
def estimate_snr(signal_rx, symbols_tx, gray_symbols):
    """
    Estimate the signal-to-noise ratio from received and known transmitted symbols.

    Parameters
    ----------
    signal_rx : array_like
        received signal
    symbols_tx : array_like
        transmitted symbol sequence
    gray_symbols : array_like
        gray coded symbols

    Note
    ----
    signal_rx and symbols_tx need to be synchronized and have the same length.

    Returns
    -------
    snr : float
        estimated linear signal-to-noise ratio
    S0 : float
        estimated linear signal power
    N0 : float
        estimated linear noise power
    """
    assert signal_rx.shape[0] >= symbols_tx.shape[0]
    N = gray_symbols.shape[0]
    L = signal_rx.shape[0]
    in_pow = 0.
    N0 = 0.
    #omp parallel for reduction(+:N0,in_pow)
    for ind in range(N):
        sel_symbs = signal_rx[symbols_tx == gray_symbols[ind]]
        K = sel_symbs.shape[0]
        Px = K/L
        mu = np.mean(sel_symbs)
        #dif = sel_symbs-mu
        sigma = np.sqrt(np.sum(abs(sel_symbs-mu)**2)/K)
        N0 += abs(sigma)**2*Px
        in_pow += abs(mu)**2*Px
    snr = in_pow/N0
    return snr, in_pow, N0

#pythran export cal_mi_mc(complex128[], complex128[], float64):
def cal_mi_mc(noise, symbols, N0):
    M = symbols.size
    L = noise.size
    mi_out = 0
    #omp parallel for reduction(+:mi_out) collapse(2)
    for i in range(M):
        for l in range(L):
            tmp = 0
            for j in range(M):
                tmp += np.exp(-(abs(symbols[i] - symbols[j])**2 + 2*np.real((symbols[i]-symbols[j])*noise[l]))/N0)
            mi_out += np.log2(tmp)
    return np.log2(M) - mi_out/M/L

#pythran export cal_mi_mc_fast(complex128[], complex128[], complex128[], float64):
def cal_mi_mc_fast(sig, sig_tx, symbols, N0):
    M = symbols.size
    L = sig.size
    mi_out = 0
    #omp parallel for reduction(+:mi_out)
    for l in range(L):
        tmp = 0
        for j in range(M):
            tmp += np.exp(-(abs(sig[l]-symbols[j])**2 - abs(sig[l]-sig_tx[l])**2)/N0)
        mi_out += np.log2(tmp)
    return np.log2(M) - mi_out/L
