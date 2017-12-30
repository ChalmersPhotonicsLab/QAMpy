# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:54:36 2017

@author: mazurm
"""

import numpy as np
#from dsp import modulation
from . import modulation
from .dsp_cython import soft_l_value_demapper as soft_l_value_demapper_pyx
from numba import jit

def generate_bitmapping_mtx(M):
    mod = modulation.QAMModulator(M)
    out_mtx = np.reshape(mod.decode(mod.gray_coded_symbols),(M,mod.bits))
    symbs = mod.gray_coded_symbols
    bit_map = np.zeros([mod.bits,int(M/2),2],dtype=complex)
    
    for bit in range(mod.bits):
        bit_map[bit,:,0] = symbs[~out_mtx[:,bit]]
        bit_map[bit,:,1] = symbs[out_mtx[:,bit]]
    return symbs, bit_map

@jit(nopython=True)
def soft_l_value_demapper(rx_symbs, M, snr,bits_map):
    #rx_symbs = np.atleast_2d(rx_symbs)
    
    # Allocate output
    num_bits = int(np.log2(M))
    L_values = np.zeros((rx_symbs.shape[0]*num_bits))
    
    # Demapp all modes
    #for mode in range(rx_symbs.shape[0]):
    for bit in range(num_bits):
        for symb in range(rx_symbs.shape[0]):
            L_values[symb*num_bits + bit] = np.log(np.sum(np.exp(-snr*np.abs(bits_map[bit,:,1]-rx_symbs[symb])**2)))-\
                np.log(np.sum(np.exp(-snr*np.abs(bits_map[bit,:,0]-rx_symbs[symb])**2)))
    return L_values

def calc_gmi(rx_symbs, tx_symbs, M):  
    rx_symbs = np.atleast_2d(rx_symbs)
    tx_symbs = np.atleast_2d(tx_symbs)
    symbs, bit_map = generate_bitmapping_mtx(M)
    
    mod = modulation.QAMModulator(M)
    num_bits = int(np.log2(M))
    GMI = np.zeros(rx_symbs.shape[0])
    GMI_per_bit = np.zeros((rx_symbs.shape[0],num_bits))
    SNR_est = np.zeros(rx_symbs.shape[0])
    # For every mode present, calculate GMI based on SD-demapping
    for mode in range(rx_symbs.shape[0]):
        print("mode %d"%mode)

        # GMI Calc
        #rx_symbs[mode] = rx_symbs[mode]/np.sqrt(np.mean(np.abs(rx_symbs[mode])**2))
        #snr = estimate_snr(rx_symbs[mode],tx_symbs[mode],symbs)[0]
        rx_symbs[mode] = rx_symbs[mode] / np.sqrt(np.mean(np.abs(rx_symbs[mode]) ** 2))
        tx, rx = mod._sync_and_adjust(tx_symbs[mode],rx_symbs[mode])
        snr = estimate_snr(rx, tx, symbs)[0]
        SNR_est[mode] = snr

        #l_values = soft_l_value_demapper(rx_symbs[mode],M,10**(snr/10),bit_map)
        #bits = mod.decode(mod.quantize(tx_symbs[mode])).astype(np.int)
        l_values = soft_l_value_demapper_pyx(rx,M,10**(snr/10),bit_map)
        bits = mod.decode(mod.quantize(tx)).astype(np.int)
        
        # GMI per bit
        for bit in range(num_bits):
            print(bit)
            GMI_per_bit[mode,bit] = 1 - np.mean(np.log2(1+np.exp(((-1)**bits[bit::num_bits])*l_values[bit::num_bits])))
        # Sum GMI
        GMI[mode] = np.sum(GMI_per_bit[mode])
    return GMI, GMI_per_bit, SNR_est

@jit
def estimate_snr(rx_symbs, tx_symbs,symbs):
    M = symbs.shape[0]
    rx_symbs = rx_symbs / np.sqrt(np.mean(np.abs(rx_symbs)**2))
    
    Px = np.zeros(M)
    N0 = 0
    mus = np.zeros(M, dtype = complex)
    sigmas = np.zeros(M, dtype= complex)
    in_pow = 0
    
    for ind in range(M):
        sel_symbs = rx_symbs[np.bool_(tx_symbs == symbs[ind])]
        Px[ind] = sel_symbs.shape[0] / rx_symbs.shape[0]
        mus[ind] = np.mean(sel_symbs)
        sigmas[ind] = np.std(sel_symbs)

        N0 += np.abs(sigmas[ind])**2*Px[ind]
        in_pow += np.abs(mus[ind])**2*Px[ind]
    
    SNR = 10*np.log10(in_pow/N0)       
    return SNR, Px, mus, in_pow, N0
