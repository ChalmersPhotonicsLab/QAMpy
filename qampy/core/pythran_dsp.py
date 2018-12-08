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
    """
    Blind phase search using Python. This is slow compared to the cython and arrayfire methods and should not be used.
    """
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

#pythran export soft_l_value_demapper(complex128[], int, float64, complex128[][][])
#pythran export soft_l_value_demapper(complex64[], int, float32, complex64[][][])
def soft_l_value_demapper(rx_symbs, M, snr, bits_map):
    num_bits = int(np.log2(M))
    L_values = np.zeros(rx_symbs.shape[0]*num_bits)
    N = rx_symbs.shape[0]
    k = bits_map.shape[1]
    #omp parallel for
    for idx in range(N*num_bits): # collapse the loop manually
        bit = idx//N
        symb = idx%N
        tmp = 0
        tmp2 = 0
        for l in range(k):
            tmp += np.exp(-snr*abs(bits_map[bit,l,1] - rx_symbs[symb])**2)
            tmp2 += np.exp(-snr*abs(bits_map[bit,l,0] - rx_symbs[symb])**2)
        L_values[idx] = np.log(tmp) - np.log(tmp2)
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

