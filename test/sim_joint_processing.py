#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:08:00 2018

@author: mazurm
"""
import sim_pilot_txrx
import numpy as np

#==============================================================================
# Do Some plotting
#==============================================================================
#

#ch_sep_array = np.arange(1.3,0.8,-.02)
num_avg = 3

snr_test = np.arange(45,15,-2)
beta_test =  np.arange(0.01,.3,.01)
Rs = np.arange(24,26.8,.2)

Ntaps=45
M=64
resBits_rx = 5
resBits_tx = 5
rx_filter_bw = 1.4

#snr_test = np.arange(45,15,-2)
#beta_test =  np.arange(0.0,.3,.01)
#Rs = np.arange(24.0,26.6,.2)

#beta_test = np.array([0.01,0.03,0.05,0.1,0.2,0.5])
gmi_res_joint = np.zeros([Rs.shape[0],snr_test.shape[0],beta_test.shape[0]])
gmi_res_single= np.zeros([Rs.shape[0],snr_test.shape[0],beta_test.shape[0]])
ber_res_joint = np.zeros([Rs.shape[0],snr_test.shape[0],beta_test.shape[0]])
ber_res_single= np.zeros([Rs.shape[0],snr_test.shape[0],beta_test.shape[0]])


for r in range(Rs.shape[0]):
    for s in range(snr_test.shape[0]):
        for b in range(beta_test.shape[0]):
            gmi_joint_tmp = np.zeros(num_avg)
            gmi_single_tmp = np.zeros(num_avg)
            ber_joint_tmp = np.zeros(num_avg)
            ber_single_tmp = np.zeros(num_avg)
            for n in range(num_avg):
                gmi1,gmi2,ber1,ber2 = sim_pilot_txrx.sim_joint_eq(Rs[r], beta=beta_test[b],sig_snr=snr_test[s],
                                                                  M=M,Ntaps=Ntaps,rx_filter_bw=rx_filter_bw, 
                                                                  resBits_rx=resBits_rx, resBits_tx=resBits_tx)
                gmi_joint_tmp[n] = np.sum(gmi1)
                gmi_single_tmp[n] = np.sum(gmi2)
                ber_joint_tmp[n] = np.mean(ber1)
                ber_single_tmp[n] = np.mean(ber2)
            gmi_res_joint[r,s,b] = np.mean(gmi_joint_tmp)
            gmi_res_single[r,s,b] = np.mean(gmi_single_tmp)
            ber_res_joint[r,s,b] = np.mean(ber_joint_tmp)
            ber_res_single[r,s,b] = np.mean(ber_single_tmp)
            print("Rs: %2.1f, SNR: %d, Beta: %1.2f, GMI-Joint: %2.2f, GMI-Ind. %2.2f"%
                  (Rs[r], snr_test[s],beta_test[b],gmi_res_joint[r,s,b],gmi_res_single[r,s,b]))

if resBits_rx is not None:
    np.savez("Sim_JointEq_%dQAM_NumTaps_%d_RxBW_%1.1f_DetailedSweeps_Enob_%1.1f"%(M,Ntaps,rx_filter_bw,resBits_rx),Rs=Rs,snr_test=snr_test,beta_test=beta_test,gmi_res_joint=gmi_res_joint,
             gmi_res_single=gmi_res_single,ber_res_joint=ber_res_joint,ber_res_single=ber_res_single,Ntaps=Ntaps,
             resBits_rx=resBits_rx,resBits_tx=resBits_tx)
else:
    np.savez("Sim_JointEq_%dQAM_NumTaps_%d_RxBW_%1.1f_DetailedSweeps"%(M,Ntaps,rx_filter_bw),Rs=Rs,snr_test=snr_test,beta_test=beta_test,gmi_res_joint=gmi_res_joint,
         gmi_res_single=gmi_res_single,ber_res_joint=ber_res_joint,ber_res_single=ber_res_single,Ntaps=Ntaps,
         resBits_rx=resBits_rx,resBits_tx=resBits_tx)
