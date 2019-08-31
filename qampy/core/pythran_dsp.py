import numpy as np

def cabsq(x):
    return x.real**2 + x.imag**2

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
    L = E.shape[0]
    p = testangles.shape[0]
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

def l_values(btx, rx, snr):
    tmp = 0
    tmp2 = 0
    k = btx.shape[0]
    for l in range(k):
        tmp += np.exp(-snr*abs(btx[l,1] - rx)**2)
        tmp2 += np.exp(-snr*abs(btx[l,0] - rx)**2)
    return tmp, tmp2

#pythran export soft_l_value_demapper(complex128[], int, float64, complex128[][][])
#pythran export soft_l_value_demapper(complex64[], int, float32, complex64[][][])
def soft_l_value_demapper(rx_symbs, num_bits, snr, bits_map):
    N = rx_symbs.shape[0]
    L_values = np.zeros((N, num_bits))
    k = bits_map.shape[1]
    #omp parallel for collapse(2)
    for symb in range(N):
        for bit in range(num_bits):
            tmp, tmp2 = l_values(bits_map[bit,:,:], rx_symbs[symb], snr)
            L_values[symb, bit] = np.log(tmp) - np.log(tmp2)
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
def select_angles(angles, idx):
    L = angles.shape[0]
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

#pythran export cal_gmi_mc_omp(complex128[], float64, int, complex128[][][])
def cal_gmi_mc_omp(symbols, snr, ns, bit_map):
    M = symbols.size
    nbits = int(np.log2(M))
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
