#import cProfile
import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation, utils, phaserecovery, processing
from timeit import default_timer as timer
import time
import multiprocessing
import zmq



fb = 40.e9
os = 1
fs = os*fb
N = 3*10**5
M = 64
QAM = modulation.QAMModulator(M)
snr = 30
lw_LO = np.linspace(10e1, 1000e1, 4)
#lw_LO = [100e3]
sers = []
Nworkers = 4
dealer = "tcp://*"
worker = "tcp://localhost"

def start_worker(url):
    con = zmq.Context()
    w = processing.PhRecWorker(url, context=con)
    w.run()

cc = zmq.Context()
dd = processing.DataDealer(dealer, context=cc)
port = dd.port
wurl = worker+":{}".format(port)

for i in range(Nworkers):
    process = multiprocessing.Process(target=start_worker, args=[wurl])
    process.daemon = True
    process.start()

X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)

xx = {}
for lw in lw_LO:
    shiftN = np.random.randint(-N/2, N/2, 1)
    xtest = np.roll(symbolsX[:(2**15-1)], shiftN)
    pp = utils.phase_noise(X.shape[0], lw, fs)
    XX = X*np.exp(1.j*pp)
    msg = {'id':lw, "data": XX, "Mtestangles": 64, "symbols": QAM.symbols, "N": 14} 
    dd.send_msg(b"do_phase_rec", msg)
    xx[lw] = xtest
    #recoverd_af,ph= phaserecovery.blindphasesearch(XX, 64, QAM.symbols, 14, method="af")
    #ser,s,d = QAM.calculate_SER(recoverd_af, symbol_tx=xtest)
    #ser2,s,d = QAM.calculate_SER(recoverd_pyx, symbol_tx=xtest)
    #ser3,s,d = QAM.calculate_SER(recoverd_2s, symbol_tx=xtest)
    #print("1 stage af ser=%g"%ser)
    #print("1 stage pyx ser=%g"%ser2)
    #print("2 stage pyx ser=%g"%ser3)
    #print("time af %.1f"%abs(t2-t1))
    #print("time pyx %.1f"%abs(t3-t2))
    #print("time 2 stage %.1f"%abs(t4-t3))
    #sers.append(ser)
for i in range(len(lw_LO)):
    results = dd.recv_msg()
    print(results['id'])
    xtest = xx[results['id']]
    ser,s,d = QAM.calculate_SER(results['Eout'], symbol_tx=xtest)
    print(ser)
#plt.plot(lw_LO, sers)
#plt.show()


