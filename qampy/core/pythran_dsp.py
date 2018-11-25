import numpy as np

#pythran export bps(complex128[], float64[][], complex128[], int)
#pythran export bps(complex64[], float32[][], complex64[], int)
def bps(E, angles, symbols, N):
    """
    Blind phase search using Python. This is slow compared to the cython and arrayfire methods and should not be used.
    """
    EE = E[:,np.newaxis]*np.exp(1.j*angles)
    idx = np.zeros(E.shape[0], dtype=int)
    dist = np.min(abs(EE[:, :, np.newaxis]-symbols)**2, axis=2)
    csum = np.cumsum(dist, axis=0)
    mvg = csum[2*N:]-csum[:-2*N]
    idx[N:-N] = mvg.argmin(1)
    return idx

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


