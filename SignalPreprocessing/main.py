import sys
import os
import re
import numpy as np
import scipy.signal as sig
import pdb
import matplotlib.pyplot as plt


from get_nadris import *
from get_normalizedPPg import *
from get_pfeatures import *
from get_sigSlope import *
from out_feature import *
from signalInterp import *
from get_Heartrate import *
from get_Spo2 import *


import sys
import os
import re
import numpy as np
import scipy.signal as sig
import pdb
import matplotlib.pyplot as plt
import pandas as pd
# region Function_use


def out_features(f, val):
    # vIR, vRD = val[0], val[1]  ### Change 3
    vIR, vRD = val[0], val[0]
    n = len(vIR[0])
    for i in range(n):
        print(f"SI{i+1:03d}A  ", end='', file=f)
        if vIR[0][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vIR[0][i]:17.8E}", file=f)
        print(f"SI{i+1:03d}V  ", end='', file=f)
        if vIR[1][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vIR[1][i]:17.8E}", file=f)

        print(f"SR{i+1:03d}A  ", end='', file=f)
        if vRD[0][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vRD[0][i]:17.8E}", file=f)
        print(f"SI{i+1:03d}V  ", end='', file=f)
        if vRD[1][i] == 'n/a':
            print(f"{'n/a':17s}", file=f)
        else:
            print(f"{vRD[1][i]:17.8E}", file=f)


def PPG_aveerr(n, pnum, label, pulse):
    pdata = []
    if pnum == 'NULL':
        for i in range(n):
            # print(f"This is I:{i}")
            print(f"i , label : {i, label}")
            pdata.append(pulse[i][label])
    else:
        for i in range(n):
            # print(f"PNUM{pnum}")
            # print(f"i , label : {i, label}")
            # print(f"PDATA {pdata}")
            pdata.append(pulse[i][pnum][label])
    # pdb.set_trace()
    print(f"pdata is: {pdata}")
    print(len(pdata))
    pdata = np.asarray(pdata)
    ave = np.mean(pdata)
    err = np.std(pdata)/np.sqrt(n-1) if n > 1 else 0.0
    return ave, err


rawsig = pd.read_csv(
    "/home/mhosein/Documents/GluCheck/GluProject/sample_20.csv")


def get_peak_feature(sig, t, i0, it, ii, w0, a):
    res = {
        't':      0,
        'pval':   0,
        'curv':   0,
        'slopeL': 0,
        'slopeR': 0
    }
    nT = len(t)
    if it+ii >= nT:
        return res

    t0 = t[it+ii]
    t1 = t[it+ii-w0] if it+ii >= w0 else t[it]
    t2 = t[it+ii+w0] if it+ii+w0 < nT else t[nT-1]
    A = sig[i0+ii]
    a4, a3, a2, a1 = a[0], a[1], a[2], a[3]

    res['t'], res['pval'] = t0, A
    res['curv'] = 2.0*a2 / np.sqrt(1.0+a1*a1)**3

    return res


def analy_peak_signal(signal, i0, iS, iN, iDN):
    imax, imin, immn = iS-i0, iS-i0, iS-i0
    smax, smin, smmn = signal[iS], signal[iS], signal[iS]
    nsig = signal.size
    if (iN >= nsig):
        iN = nsig

    for i in range(iS+1, iN):
        if i < iDN:
            if signal[i] > smax:
                imax = i-i0
                imin = i-i0

    return imax, imin, immn


def get_dA1_peak(para, dA1):
    ndA1 = dA1.size
    pmax = dA1[0:int(ndA1/4)+1].max()
    imax = dA1[0:int(ndA1/4)+1].argmax()
    pmin2 = dA1[imax:int(ndA1/3)+1].min()
    imin2 = dA1[imax:int(ndA1/3)+1].argmin() + imax
    v_ave = para['dA1_plat_err'] * pmax

    ip1, ip2, i0arr, iNarr = 0, 0, [], []

    res = {
        'pmax':  pmax,
        'imax':  imax,
        'imin2': imin2,
        'ip1':   ip1,
        'ip2':   ip2,
    }
    return res


def fit_dA1_peak(para, f, key, t, dA1, res):
    print(f"x0 = {t[res['imax']]:f}",           file=f)
    print(f"s  = {t[res['imax']]/2.0:f}",       file=f)
    print(f"a  = {res['pmax']:f}",              file=f)
    print(f"fit f(x) '-' using 1:2 via a,x0,s", file=f)
    i0 = int(res['imax']/2)
    iN = i0 + res['imax']
    for i in range(i0, iN):
        print(f"{t[i]:f}  {dA1[i]:f}", file=f)
    print(f"e", file=f)
    print(f"print \"{key} \", a, x0, s", file=f)


def get_DNotch_1(t, pulse):
    dt = t[1] - t[0]
    di = int(0.015/dt + 0.5)
    if di <= 0:
        di = 1

    iDPs, vDPs, iDNs, vDNs = [], [], [], []
    while True:
        iDP, vDP, iDN, vDN = 0, pmin, 0, pmax
        for i in range(i1+di, iN-di):
            if sig[i] > sig[i-di] and sig[i] > sig[i+di] and sig[i] > vDP:
                iDP, vDP = i, sig[i]
#        iN2 = np.maximum(i1+di, iN-di) if iDP <= 0 else iDP
        iN2 = iN-di if iDP <= 0 else iDP
        for i in range(i2+di, iN2):
            if sig[i] < sig[i-di] and sig[i] < sig[i+di] and sig[i] < vDN:
                iDN, vDN = i, sig[i]
        iDPs.append(iDP)
        vDPs.append(vDP)
        iDNs.append(iDN)
        vDNs.append(vDN)
        if iDP <= 0:
            break
        i1 = iDP + int(0.1/dt + 0.5)
        i2 = iDP

    pulse['iDN'] = iDNs[k]+pi0
    pulse['iDP'] = iDPs[k]+pi0
    pulse['DNval'] = sig[iDNs[k]] - pmin
    pulse['DPval'] = sig[iDPs[k]] - pmin
    pulse['tDNval'] = t[iDNs[k]+pi0] - t[pi0]
    pulse['tDPval'] = t[iDPs[k]+pi0] - t[pi0]
    return True


def get_DNotch_2(t, pulse):
    sig, dA1, pi0, piM = pulse['sAC'], pulse['dA1'], pulse['i0'], pulse['iM2']
    dt = t[1]-t[0]
    pmin = sig[0]
    i0 = piM-pi0+1
    iN = int(dA1.size * 2/3)

    jNs, vNs = [], []
    while True:
        stat, jN, vN = 0, 0, 0
        for n in range(i0, iN):
            if dt*(n-i0) < 0.075:
                continue
            dA2 = dA1[n+1] - dA1[n]
            if stat == 0:
                if dA2 < 0:
                    continue
                stat, jN, vN = 1, n, dA2
            else:
                if dA2 < 0 and dt*(n-jN) > 0.05:
                    break
                if dA2 > vN:
                    jN, vN = n, dA2
        jNs.append(jN)
        vNs.append(vN)
        if jN == 0:
            break
        i0 = jN + int(0.05/dt + 0.5)

    vNmax, k = jNs[0], 0
    for n in range(len(jNs)):
        if vNs[n] > 0.5:
            k = n
            break
        if vNmax < vNs[n]:
            k, vNmax = n, vNs[n]
    i0, stat = jNs[k]+1, 0

    pulse['iDN'] = iDN
    pulse['iDP'] = iDP
    pulse['DPval'] = sig[iDP-pi0] - pmin
    pulse['DNval'] = sig[iDN-pi0] - pmin
    pulse['tDPval'] = t[iDP] - t[pi0]
    pulse['tDNval'] = t[iDN] - t[pi0]


def get_TailSlope(t, pulse):
    sig, ipeak, slopR = pulse['sAC'], pulse['pulse_peak'], 0.0
    i0 = ipeak[-1] - pulse['i0']
    iN = sig.size
    di = int((iN-i0)/6.0 + 0.5)
    if di > 1:
        for j in range(1, 6):
            i = i0 + di*j
            i1 = i-3 if i-3 > i0 else i0
            i2 = i+3 if i+3 < iN else iN-1
            slopR = slopR + (sig[i2]-sig[i1]) / (t[i2]-t[i1])
    pulse['TailSlope'] = slopR / 5.0


def get_all_minmax(para, dt, nT, sig, i0, iN):
    """ Bigger Fall Side Detection algorighm """
    lmax, lmax2, lmin = [], [], []
    tx,   px,    dp = [], [], []
    for i in range(1, nT-1):
        if (sig[i]-sig[i-1])*(sig[i+1]-sig[i]) < 0:
            tx.append(i)
            px.append(sig[i])
    np = len(tx)

    for i in range(np-1):
        r = 0 if px[i] >= px[i+1] else px[i+1]-px[i]
        dp.append(r)
    d2 = dp.copy()
    d2.sort(reverse=True)
    dm = d2[20]

    TH_Ratio = para['TH_Ratio']
    for i in range(len(dp)):
        if dp[i] < dm*TH_Ratio or dp[i] > dm/TH_Ratio:
            continue
        if sig[tx[i]] < 0.0 and sig[tx[i+1]] > 0.0:
            lmin.append(tx[i])
            lmax.append(tx[i+1])

    return lmax, lmin


def ppg_ac_normalize(sAC):
    ave = np.mean(sAC)
    err = np.std(sAC)
    sAC = (sAC - ave) / err
    print(f"ave is: {ave}")
    # ave is: 0.4774533817821295
    print(f"err is: {err}")
    # err is: 41.08309507224383
    return sAC


def search_t_idx(nT, t, t0):
    if (t[0] >= t0):
        return 0
    if (t[nT-1] <= t0):
        return nT-1

    i0 = int(nT/2)
    di = int(nT/4)
    while (t[i0] > t0 or t[i0+1] <= t0) and di > 0:
        i0 = i0-di if (t[i0] > t0) else i0+di
        di = int(di/2)
    if t[i0+1] <= t0:
        while i0 < nT:
            if t[i0] <= t0 and t[i0+1] > t0:
                break
            i0 = i0+1

    else:
        return 0 if i0 < 0 else nT-1


def InterpSimple(x0, f0, x1):
    fg = []
    k = 0
    n = len(x0)
    for i in range(len(x1)):
        while k < n-1 and x0[k] < x1[i]:
            k = k+1
        k = k-1
        r = f0[k] + (f0[k+1]-f0[k]) * (x1[i]-x0[k]) / (x0[k+1]-x0[k])
        fg.append(r)
    return fg


def InterpDataNorm(x0, f0):
    ff0 = np.asarray(f0, dtype=float)
    xx0 = np.asarray(x0, dtype=float)
    fmax = np.abs(ff0).max()
    ff0 = ff0 / fmax

    dx = xx0[1] - xx0[0]
    xscl = 1
    while dx < 0.1:
        xscl = xscl * 10
        dx = dx * 10
    while dx > 1.0:
        xscl = xscl / 10
        dx = dx / 10
    x0R = xx0 * xscl
    return fmax, x0R, ff0


def InterpAkima(x0, f0, x1):
    n = len(x0)
    k = 0
    fg = []
    ss = np.zeros(n+3, dtype=float)
    f1 = np.zeros(n, dtype=float)
    c0 = np.zeros(n, dtype=float)
    c1 = np.zeros(n, dtype=float)
    c2 = np.zeros(n, dtype=float)
    c3 = np.zeros(n, dtype=float)
    fscl, x0R, f0R = InterpDataNorm(x0, f0)
    for i in range(n-1):
        ss[i+2] = (f0R[i+1] - f0R[i]) / (x0R[i+1] - x0R[i])
    ss[1] = 2.0*ss[2] - ss[3]
    ss[0] = 2.0*ss[1] - ss[2]
    ss[n+1] = 2.0*ss[n] - ss[n-1]
    ss[n+2] = 2.0*ss[n+1] - ss[n]

    for i in range(len(x1)):
        while k < n-1 and x0[k] < x1[i]:
            k = k+1
        k = k-1
        if k > n-1:
            break
        u = (x1[i] - x0[k]) / (x0[k+1] - x0[k])
        r = (c0[k] + c1[k]*u + c2[k]*u*u + c3[k]*u*u*u)*fscl
        fg.append(r)
    return fg


def SignalInterp(para, x0, f0, x1):

    if (para['interp_algo']) == 1:
        return InterpSimple(x0, f0, x1)
    elif (para['interp_algo']) == 2:
        return InterpAkima(x0, f0, x1)
    else:
        print(f"!!! Unknown signal interpolation: {para['interp_algo']}")
        exit(1)


# endregion
######################################################
# region Parameter
para = {
    'outdir':       'out',
    'ttot':         30,
    'interp_algo':  2,
    'dsample':      300,

}

para['ttot'] -= para['T_init_skip']

para['toff'] = 30
para['w0'] = 15
para['w1'] = 10

print(len(para))
print(para)
# endregion
# --------------------------------------------------------------------------

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <sub_info filename>")
    exit(0)
fn = sys.argv[1]
# pdb.set_trace()
# print(fn)
try:
    f = open(fn, 'r')
except:
    print(f"!!! read_subinfo: cannot open file: {fn}")
    exit(1)
lines = f.readlines()
label = os.path.split(os.path.split(fn)[0])[1]
f.close()

print(lines)
print(label)
# region	Read and check sub_info file.
Ni, Li, Ri, Hi, gi = -1, -1, -1, -1, -1
d = {}
info = {}
for li, line in enumerate(lines):
    line = line.rstrip()
    print(line.find('Index'))
    if line.find('Index') == 0:
        for i, v in enumerate(line.lower().split()):
            if v == 'num':
                Ni = i
                print(f"Ni: {Ni}")
            if v == 'lppg_path':
                Li = i
                print(f"Li: {Li}")
            if v == 'rppg_path':
                Ri = i
                print(f"Ri: {Ri}")
            if v == 'height':
                Hi = i
                print(f"Hi: {Hi}")
            if v == 'gender':
                gi = i
                print(f"gi: {gi}")
    elif len(line) > 0 and line[0].isdigit() == True:
        # pdb.set_trace()
        arr = line.split()
        print(f"Arr is: {arr}")
        # Arr is: ['1', '112233', '1_113_105_1', '1_113_105', '177', '1']
        print(f"Label is: {label}")
        # 1
        print(f"Fn is: {fn}")
        # Fn is: /home/mhosein/Documents/GluCheck/DL_GLU/raw_samp/1/ppg.txt
        if Ni >= 0 or Li >= 0 or Ri >= 0 or Hi >= 0 or gi >= 0:
            d['label'] = label
            d['i_fn'] = fn
            d['ok'] = 'NULL'
            d['Pid'] = int(arr[Ni])
            d['L_fn'] = arr[Li]
            d['R_fn'] = arr[Ri]
            d['Height'] = float(arr[Hi])
            d['gender'] = int(arr[gi])
            d['lineNo'] = li
            info = d.copy()
            print(f"Info is: {info}")
            # Info is: [{'label': '1', 'i_fn': '/home/mhosein/Documents/GluCheck/DL_GLU/raw_samp/1/ppg.txt', 'ok': 'NULL', 'Pid': 112233, 'L_fn': '1_113_105_1', 'R_fn': '1_113_105', 'Height': 177.0, 'gender': 1, 'lineNo': 1}]
# print(info)


# Info is: [{'label': '1', 'i_fn': '/home/mhosein/Documents/GluCheck/DL_GLU/raw_samp/1/ppg.txt', 'ok': 'NULL', 'Pid': 112233, 'L_fn': '1_113_105_1', 'R_fn': '1_113_105', 'Height': 177.0, 'gender': 1, 'lineNo': 1}]
# pdb.set_trace()
print(info)
print(info['i_fn'])
idir = os.path.split(info['i_fn'])[0]
# IDIR is : /home/mhosein/Documents/GluCheck/DL_GLU/raw_samp/1
print(f"IDIR is : {idir}")
odir = para['outdir']
dset = []
type = []
ok = {}

#### left hand ####
type = 'L4'
rawfn = info['L_fn']
name = os.path.split(rawfn)[1]
print(f"name is {name}")
# name is 1_113_105_1
label = info['label']
print(f"label is: {label}")
# label is: 1
odir = os.path.join(odir, label)
print(f"odir is: {odir}")

dFext = '_step'
# odir is: out0/1
outfn = os.path.join(odir, f"{name}{dFext}.txt")
print(f"outfn is: {outfn}")
# outfn is: out0/1/1_113_105_1_step.txt
rawfn = f"{rawfn}.csv" if os.path.exists(f"{rawfn}.csv") == True else \
    os.path.join(idir, f"{rawfn}.csv")
run = para['runPPGL']
dL = {
    'info':   info,
    'label':  label,
    'name':   name,
    'type':   type,
    'outdir': odir,
    'datafn': rawfn,
    'outfn':  outfn,
    'outfs':  [],
    'run':    run,
    'n_out':  0,
    'sigok':  0,
}
print(f"Left hand dset is: {dL}")
# endregion


dset = {}
type = []
ok = {}
#### This part append the data of left and right hand to dset####

for dd in [dL]:
    if dd['run'] != 'yes':
        continue

    if os.path.exists(dd['outfn']) == False:
        dset = dd
        ok[dd['type']] = 0
    else:
        ok[dd['type']] = -1
    type.append(dd['type'])
if len(dset) > 0:
    os.makedirs(dL['outdir'], exist_ok=True)
    os.makedirs(para['wdir'], exist_ok=True)

print(f"ok is: {ok}")

print(f"type is: {type}")

info['type'] = type.copy()
info['ok'] = ok.copy()


# --------------------------------------------------------------------------
# Read the source data.

sigIR0, t0 = [], [],
sigIR, t = [], [],
t_skip = para['T_init_skip']

try:
    f = open(dset['datafn'], 'r')
except:
    print("Helooooo")


lines = f.readlines()
print(f"lines is: {lines}")
f.close()

for line in lines[1:]:
    line = line.rstrip().lstrip()
    arr = line.split(',')
    sigIR0.append(float(arr[0]))


print(f"sigIR0 is; {sigIR0}")
nT = len(sigIR0)
print(f"the length of signal is: {nT}")


dsmpraw = 1

dsample = para['dsample']
T_tot = para['ttot'] + t_skip
print(T_tot)
print(dsample)
ts = -1
dT = T_tot / (nT-1)

for i in range(nT):
    tx = i * dT
    if tx < t_skip:
        continue
    if ts == -1:
        ts = tx
    t0.append(tx-ts)

    sigIR.append(sigIR0[i])


print(f"t0 is: {t0[-1]}")
print(f"sigIR is:{sigIR}")
nT = int(dsample * T_tot)

dT = T_tot / (nT-1)
print(f"dT is: {dT}")

for i in range(nT-1):
    tx = i * dT
    if tx >= t_skip:
        t.append(tx - t_skip)

dset['sigIR'] = SignalInterp(para, t0, sigIR, t)


plt.title("Raw Signal")
plt.plot(rawsig)
plt.figure()
plt.title("Filtered Signal")
plt.plot(t, dset['sigIR'])
plt.show()

dset['nT'] = len(t)
dset['t'] = t
dset['dsmpraw'] = dsmpraw
#


####### end of read source region ########
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
######### ppg sig analy ##########

ok = False
res = []

# region Analyze_ppg_signal


print("* Analyze PPG signals ....")
for signame in ['IR']:  # Change  for signame in ['IR', 'RD']:
    # pdb.set_trace()
    nT = dset['nT']
    print(f"nT is:{nT}")
    # nT is:14249
    t = dset['t']
    print(f"t is: {t}")
    # [..., 56.81598773251551, 56.81998799919995, 56.823988265884395, 56.827988532568845, 56.83198879925329, 56.83598906593774, 56.83998933262218, 56.843989599306624, 56.847989865991075, 56.85199013267552, 56.85599039935996, 56.85999066604441, 56.86399093272885, 56.867991199413304, 56.87199146609775, 56.87599173278219, 56.87999199946664, 56.88399226615108, 56.887992532835526, 56.891992799519976, 56.89599306620442, 56.89999333288886, 56.90399359957331, 56.907993866257755, 56.911994132942205, 56.91599439962665, 56.91999466631109, 56.92399493299554, 56.927995199679984, 56.93199546636443, 56.93599573304888, 56.93999599973332, 56.94399626641777, 56.94799653310221, 56.951996799786656, 56.955997066471106, 56.95999733315555, 56.96399759983999, 56.96799786652444, 56.971998133208885, 56.975998399893335, 56.97999866657778, 56.98399893326222, 56.98799919994667, 56.991999466631114, 56.99599973331556]
    t0 = para['T_dt_stepb']
    print(f" t0 is: {t0}")
    # t0 is: 0.0
    # t0 is: 0.1
    tN = para['T_step']
    print(f"tN is: {tN}")
    # tN is: 60.0
    # tN is: 30.0
    i0 = search_t_idx(nT, t, t0)
    print(f"is: {i0}")
    # i0 is: 0
    # i0 is: 2
    iN = search_t_idx(nT, t, tN)
    print(f"iN is: {iN}")

    # iN is: 14248
    # iN is: 8098
    if i0 == -1 or iN == -1:
        print(
            f"!!! PPG_preproc: cannot find t-idx: t0=[{t0:f},{tN:f}], idx=[{i0}:{iN}]")
        exit(1)
    # ok, res = sep_AC_DC(para, data, signame, i0, iN)
    order = 6
    ok = True
    t = dset['t']
    print(f"t is:{t}")
    # [..., 56.81598773251551, 56.81998799919995, 56.823988265884395, 56.827988532568845, 56.83198879925329, 56.83598906593774, 56.83998933262218, 56.843989599306624, 56.847989865991075, 56.85199013267552, 56.85599039935996, 56.85999066604441, 56.86399093272885, 56.867991199413304, 56.87199146609775, 56.87599173278219, 56.87999199946664, 56.88399226615108, 56.887992532835526, 56.891992799519976, 56.89599306620442, 56.89999333288886, 56.90399359957331, 56.907993866257755, 56.911994132942205, 56.91599439962665, 56.91999466631109, 56.92399493299554, 56.927995199679984, 56.93199546636443, 56.93599573304888, 56.93999599973332, 56.94399626641777, 56.94799653310221, 56.951996799786656, 56.955997066471106, 56.95999733315555, 56.96399759983999, 56.96799786652444, 56.971998133208885, 56.975998399893335, 56.97999866657778, 56.98399893326222, 56.98799919994667, 56.991999466631114, 56.99599973331556]
    nT = dset['nT']
    print(f"nT is: {nT}")
    # nT is: 14249
    # nT is: 8099
    v = np.asarray(dset[f"sig{signame}"], dtype=float)
    # print(f"v is: {v}")

    # plot

    F = int(1.0/((t[nT-1] - t[0]) / (nT-1)) + 0.5)
    print(f"F is {F}")
    # F is 250
    # F is 300
    res = {'t': t, 'nT': nT, 'sig': v}
    print(f"res is {res}")
    # should write this to a log file
    # 56.83998933262218, 56.843989599306624, 56.847989865991075, 56.85199013267552, 56.85599039935996, 56.85999066604441, 56.86399093272885, 56.867991199413304, 56.87199146609775, 56.87599173278219, 56.87999199946664, 56.88399226615108, 56.887992532835526, 56.891992799519976, 56.89599306620442, 56.89999333288886, 56.90399359957331, 56.907993866257755, 56.911994132942205, 56.91599439962665, 56.91999466631109, 56.92399493299554, 56.927995199679984, 56.93199546636443, 56.93599573304888, 56.93999599973332, 56.94399626641777, 56.94799653310221, 56.951996799786656, 56.955997066471106, 56.95999733315555, 56.96399759983999, 56.96799786652444, 56.971998133208885, 56.975998399893335, 56.97999866657778, 56.98399893326222, 56.98799919994667, 56.991999466631114, 56.99599973331556], 'nT': 14249, 'sig': array([-35052.7633058 , -35089.45155789, -34933.83351218, ...,
    # -32217.11183117, -32217.15848963, -32216.68324228])}
    freq = para['NZ_Hz']
    print(f"freq is: {freq}")
    # freq is: 7.5
    # freq is: 10
    if freq != 0:
        b, a = sig.butter(order, 2*freq/F)
        print(f"b and a is:{b, a}")
        # b and a is:(array([4.95352235e-07, 2.97211341e-06, 7.43028353e-06, 9.90704471e-06,
        # 7.43028353e-06, 2.97211341e-06, 4.95352235e-07]), array([  1.        ,  -5.27191857,  11.61992772, -13.70269593,
        # 9.1160663 ,  -3.24341985,   0.48207203]))
        npad = 3 * (np.maximum(b.size, a.size) - 1)
        print(f"npad is: {npad}")
        # npad is: 18
        v = sig.filtfilt(b, a, v, padtype='odd', padlen=npad)
        res['sig'] = v
        print(f"res[sig] is: {res['sig']}")
        # plt.plot(t, v, 'red')

        # res[sig] is: [-35052.81698175 -35020.79785049 -34989.17287377 ... -32215.13084909
        # -32215.08391609 -32215.05037192]
    freq = para['AC_Hz']
    print(f"freq is: {freq}")
    # freq is: 0.75
    b, a = sig.butter(order, 2*freq/F)
    print(f"b and a is:{b, a}")
    # b and a is:(array([6.75910179e-13, 4.05546107e-12, 1.01386527e-11, 1.35182036e-11,
    # 1.01386527e-11, 4.05546107e-12, 6.75910179e-13]), array([  1.        ,  -5.92717112,  14.63850289, -19.28223947,
    # 14.28741321,  -5.64626396,   0.92975845]))
    npad = 3 * (np.maximum(b.size, a.size) - 1)
    print(f"npad is: {npad}")
    # npad is: 18
    sDC = sig.filtfilt(b, a, v, padtype='odd', padlen=npad)
    print(f"sDC is: {sDC}")

    # plot
    # plt.figure()
    # plt.plot(t, sDC)

    # sDC is: [-35073.17493741 -35069.54467671 -35065.91286647 ... -32263.47568914
    # -32263.47568947 -32263.47568965]
    sAC = v - sDC

    # plot
    # plt.figure()
    # plt.plot(t, sAC, 'red')
    # plt.show()

    print(f"sAC is {sAC}")
    # sAC is [20.35795566 48.74682623 76.7399927  ... 48.34484004 48.39177338
    # 48.42531773]
    dt = t[1] - t[0]
    print(f"dt is: {dt}")
    # dt is: 0.004000266684445641
    nn = int(1.0 / dt)
    print(f"nn is: {nn}")
    # nn is: 249
    # nn is: 299
    j, x0, y0 = i0, [], []
    print(f"iN and nT is: {iN, nT}")
    # iN and nT is: (14248, 14249)
    # iN and nT is: (8098, 8099)

    # plot

    # plt.figure()
    # plt.plot(range(len(sAC[2:301])), sAC[2:301])
    # plt.show()
    # pdb.set_trace()
    ############ this part is just for make signal symmetric to x axis ############
    while j < iN and j < nT:
        n = nn if j+nn <= nT else nT-j

        print(n)
        lmin = sAC[j:j+n].min()
        print(lmin)
        lmax = sAC[j:j+n].max()
        print(lmax)
        print(j)
        x0.append(j)
        y0.append((lmin+lmax)/2.0)
        j = j + n
    # pdb.set_trace()
    print(f"x0 is: {x0}")
    # x0 is: [0, 249, 498, 747, 996, 1245, 1494, 1743, 1992, 2241, 2490, 2739, 2988, 3237, 3486, 3735, 3984, 4233, 4482, 4731, 4980, 5229, 5478, 5727, 5976, 6225, 6474, 6723, 6972, 7221, 7470, 7719, 7968, 8217, 8466, 8715, 8964, 9213, 9462, 9711, 9960, 10209, 10458, 10707, 10956, 11205, 11454, 11703, 11952, 12201, 12450, 12699, 12948, 13197, 13446, 13695, 13944, 14193]
    print(f"y0 is: {y0}")
    # y0 is: [106.42132094601402, -8.297303951196227, -23.19775909751479, -33.32262238066323, 49.40612286096075, 11.4766803641578, 8.677130807489448, 19.955109581500437, 6.522716729898093, -7.991177950407291, 20.7057208778715, 33.944983707231586, -3.1228875781634997, 51.88615740523528, -7.847158120242966, -34.22599961708329, -13.820011650561355, 0.3743060844062711, -0.23835368846266647, 2.39793571667542, -27.18061654205667, -17.671895881459932, 0.7651875560623012, 2.9987774428627745, 3.113102179955604, 5.887912875874463, 0.7084553386557673, -4.800965984166396, 1.1629214204913296, -15.425412571032211, -29.984551415960595, 11.354738387664838, -8.40204498710591, 15.51052076553242, -1.490266859478652, -21.721811747918764, -3.7526257516710757, 2.063388206619493, -0.6200485637018573, 0.953887234612921, 7.933380983149618, 33.41741412879492, -2.070885696550249, 1.5420733164064586, 2.2275632706259785, 1.446248659367484, 4.361632725051095, -2.0958427516216034, 25.42855171757401, 9.686061523809258, -0.784121669965316, 0.762757666529069, -1.0399125245385221, -0.38286867194801744, 2.6467510405709618, -7.778242867339941, -25.101546906738804, 39.02417053726276]
    # plt.figure()
    # plt.plot(x0, y0)
    # plt.show()
    # pdb.set_trace()
    for i in range(len(x0)-1):
        dy = (y0[i+1] - y0[i]) / (x0[i+1] - x0[i])
        k = [y0[i]+(j-x0[i])*dy for j in range(x0[i], x0[i+1])]
        k = np.asarray(k)
        sAC[x0[i]:x0[i+1]] = sAC[x0[i]:x0[i+1]] - k
        sDC[x0[i]:x0[i+1]] = sDC[x0[i]:x0[i+1]] + k
    sAC[x0[-1]:nT] = sAC[x0[-1]:nT] - y0[-1]
    sDC[x0[-1]:nT] = sDC[x0[-1]:nT] + y0[-1]

    # plot
    # print(f"sAC is: {sAC}")
    # plt.figure()
    # plt.plot(t, sAC, 'green')

# endregion
 #####################################################################################################

    print(f"sDC is: {sDC}")

    print(len(x0))
    print(len(y0))
    if para['PPG_Norm'] == 1:
        sAC = ppg_ac_normalize(sAC)
        print(f"sAC is: {sAC}")

    # plot
    plt.figure()
    plt.title("Normalized PPG")
    plt.plot(t, sAC, 'blue')
    # plt.show()
    # pdb.set_trace()
    print(nT)


# region get All minMax
    ##################### get All minMax ########################
    # """ Bigger Fall Side Detection algorighm """
    # lmax, lmax2, lmin = [], [], []
    # tx,   px,    dp   = [], [], []
    # import sys
    # for i in range(1, nT-1):
    #     if (sAC[i]-sAC[i-1])*(sAC[i+1]-sAC[i]) < 0  :
    #         tx.append(i)
    #         px.append(sAC[i])
    # np = len(tx)

    # print(len(tx))
    # # plt.figure()
    # # plt.plot(t, sAC,  [i*dt for i in tx], px, '*')
    # # plt.show()
    # # pdb.set_trace()

    # for i in range(np-1):
    #     r = 0 if px[i] >= px[i+1] else px[i+1]-px[i]
    #     dp.append(r)
    # d2 = dp.copy()
    # print(d2)
    # # pdb.set_trace()
    # d2.sort(reverse=True)
    # dm = d2[20]
    # print(dm)
    # print(d2)
    # # pdb.set_trace()
    # TH_Ratio = para['TH_Ratio']
    # TH_RATIO = 0.005
    # for i in range(len(dp)):
    #     if dp[i] < dm*TH_Ratio or dp[i] > dm/TH_Ratio: continue
    #     if sAC[tx[i]] < 0.0 and sAC[tx[i+1]] > 0.0:
    #         lmin.append(tx[i])
    #         lmax.append(tx[i+1])
    # j = 0
    # for i in range(np):
    #     if j >= len(lmax):  break
    #     if tx[i] < lmax[j]: continue
    #     if tx[i] == lmax[j]:
    #         k   = i
    #         dmm = px[i]
    #     elif j < len(lmin)-1 and tx[i] < lmin[j+1]:
    #         if dmm < px[i]:
    #             k   = i
    #             dmm = px[i]
    #     else:
    #         lmax2.append(tx[k])
    #         j = j+1

    # print(f"lmax is: {lmax}")
    # # pdb.set_trace()
    # #lmax is: [172, 325, 480, 641, 807, 964, 1122, 1279, 1443, 1605, 1763, 1920, 2075, 2241, 2413, 2583, 2748, 2906, 3071, 3239, 3404, 3568, 3729, 3899, 4073, 4246, 4415, 4572, 4730, 4895, 5067, 5236, 5398, 5557, 5715, 5880, 6054, 6231, 6403, 6572, 6738, 6916, 7085, 7259, 7430, 7601, 7772, 7971]
    # print(f"lmax2 is: {lmax2}")
    # #lmax2 is: [172, 325, 480, 641, 807, 964, 1122, 1279, 1443, 1605, 1763, 1920, 2075, 2241, 2413, 2583, 2748, 2906, 3071, 3239, 3404, 3568, 3729, 3921, 4073, 4246, 4415, 4572, 4730, 4895, 5086, 5236, 5398, 5557, 5715, 5901, 6077, 6250, 6403, 6572, 6738, 6931, 7107, 7275, 7449, 7620, 7772]
    # print(f"lmin is: {lmin}")
    # #lmin is: [133, 287, 441, 602, 765, 923, 1083, 1241, 1402, 1564, 1723, 1880, 2036, 2198, 2368, 2542, 2705, 2867, 3030, 3197, 3363, 3525, 3690, 3857, 4032, 4204, 4373, 4534, 4691, 4852, 5025, 5194, 5358, 5517, 5676, 5837, 6013, 6187, 6360, 6530, 6699, 6870, 7042, 7214, 7386, 7555, 7731, 7903]
    # print(len(lmax))
    # 48
    # ----------------------------------------------------------------------------------
    lmax, lmax2, lmin = get_all_minmax(para, dt, nT-30, sAC, i0, iN)
    if len(lmin) <= 1 or len(lmax) <= 1 or len(lmax2) <= 1:
        print(f"!!! {signame}: The signal is too noisy to find any pulse.")
        print(f"!!! {signame}: Please try to tune the following parameters:")
        print(f"!!!         DATA_SAMPLE_R, NZ_HZ, TH_RATIO.")
        print(f"!!! {signame}: ignore this signal.")
        ok = False

    res['signame'] = signame
    res['sDC'] = sDC
    res['sAC'] = sAC
    res['lmax'] = np.asarray(lmax)
    res['lmax2'] = np.asarray(lmax2)
    res['lmin'] = np.asarray(lmin)
    res['nmax'] = len(lmax)
    res['nmax2'] = len(lmax2)
    res['nmin'] = len(lmin)
    res['i0'] = i0
    res['iN'] = iN
    # pdb.set_trace()
    print(f"i0 is{i0}")
    print(f"iN is{iN}")
# endregion
    peak = []
    for index in lmax:
        # pdb.set_trace()
        # print(index)
        # print(dset['sigIR'][index])
        peak.append(dset['sigIR'][index])
    peak2 = []
    for index in lmax2:
        # pdb.set_trace()
        # print(index)
        # print(dset['sigIR'][index])
        peak2.append(dset['sigIR'][index])
    valley = []
    for index in lmin:
        # pdb.set_trace()
        # print(index)
        # print(dset['sigIR'][index])
        valley.append(dset['sigIR'][index])
    # pdb.set_trace()

    # lmax = [float(i) for i in lmax]
    lmax = np.asarray(lmax)
    lmax2 = np.asarray(lmax2)
    lmin = np.asarray(lmin)
    a = lmax*0.1
    print(f"peak is: {peak}")

    # pdb.set_trace()
    # plt.plot(t, dset['sigIR'])
    # plt.plot(lmax*dt, peak, '*')
    # plt.plot(lmax2*dt, peak2, '^')
    # plt.plot(lmin*dt, valley, '*')
    # plt.show()

    peak = []
    for index in lmax:
        # pdb.set_trace()
        # print(index)
        # print(dset['sigIR'][index])
        peak.append(sAC[index])

    # pdb.set_trace()
    # plt.figure()
    # plt.plot(t, sAC)
    # plt.plot(lmax*dt, peak, '*')
    # plt.show()

    peak = []
    for index in lmax:
        # pdb.set_trace()
        # print(index)
        # print(dset['sigIR'][index])
        peak.append(sAC[index])

    valley = []
    for index in lmin:
        # pdb.set_trace()
        # print(index)
        # print(dset['sigIR'][index])
        valley.append(sAC[index])
    # pdb.set_trace()
    plt.figure()

    plt.title("Signal With Peaks&valleys")
    plt.plot(t, sAC)
    plt.plot(lmax*dt, peak, '*')

    plt.plot(lmin*dt, valley, '^')
    plt.show()


# region get Peal features
    if ok == True:
        # pdb.set_trace()
        print(res)
        rs = res
    # print(f"    {data['label']}: {data['name']}, {rs['signame']} ....")
        print(f"rs is: {rs}")
        # pulse = sep_HB_pulse(rs)
        lmin, lmax, lmax2 = rs['lmin'], rs['lmax'], rs['lmax2']
        sAC, sDC, sACDC, t = rs['sAC'], rs['sDC'], rs['sig'], rs['t']
        nlmin, nlmax, pulse = len(lmin), len(lmax), []
        dt = t[1] - t[0]
        peak = []
        print(nlmin)
        for i in range(nlmin-1):
            pulse_peak, pulse_vally, p = [], [], {}
            ii = i if lmin[i] < lmax[i] else i+1
            i2 = i if lmin[i] < lmax2[i] else i+1
            sAC0 = sAC[lmin[i]]
            sig = [sAC[j] for j in range(lmin[i], lmin[i+1])]
            dA1 = [(sig[j+1]-sig[j])/dt for j in range(len(sig)-1)]
            aDC = sDC[lmin[i]:lmin[i+1]].mean()
            for j in range(lmin[i]+2, lmin[i+1]-1):
                if sAC[j+1] > sAC[j] and sAC[j-1] > sAC[j]:
                    pulse_vally.append(j)
                if sAC[j+1] < sAC[j] and sAC[j-1] < sAC[j]:
                    pulse_peak.append(j)
                    peak.append(j)
            p = {
                'idx':    i,
                'sAC':    np.asarray(sig),
                'dA1':    np.asarray(dA1),
                'i0':     lmin[i],
                'iN':     lmin[i+1],
                'iM':     lmax[ii],
                'iM2':    lmax2[i2],
                'Lm':     sACDC[lmin[i]],
                'LM':     sACDC[lmax2[i2]],
                'Ampval': sACDC[lmax2[i2]]-sACDC[lmin[i]],
                'aDC':    aDC,
                'pulse_vally':  pulse_vally.copy(),
                'pulse_peak':   pulse_peak.copy(),
                'npulse_vally': len(pulse_vally),
                'npulse_peak':  len(pulse_peak),
            }

            # plt.figure()
            # plt.plot(range(len(sig)), sig)
            # plt.show()
            pulse.append(p)
        # pdb.set_trace()
        sum = 0
        counter = 0
        # print(pulse['pulse_peak'])
        print(peak)

        for i in range(0, len(peak)-1):
            peak_dist = (- peak[i] + peak[i+1])*dt
            print(peak[i])
            print(peak[i+1])
            # pdb.set_trace()
            print(peak_dist)
            if 50 < (60/peak_dist):  # and (60/peak_dist) < 200:
                hr = 60 / peak_dist
                print(f"hr is: {hr}")
                sum += hr

                counter += 1

        # pdb.set_trace()
        HeartRate = sum / counter
        print(HeartRate)
        print(len(pulse))
        print(f'pulse is" {pulse}')

    lmin, lmax, lmax2, t = rs['lmin'], rs['lmax'], rs['lmax2'], rs['t']
    n1, n2, n3, j, j2 = len(lmin), len(lmax), len(lmax2), 0, 0
    print(f"n1 is: {n1}")
    # n1 is: 40
    while j < n2 and lmax[j] <= lmin[0]:
        j = j+1
    while j2 < n3 and lmax2[j] <= lmin[0]:
        j2 = j2+1
    for i in range(n1-1):
        if i+j < n2 and i+j2 < n3:
            pulse[i]['HBTval'] = t[lmin[i+1]] - t[lmin[i]]
            pulse[i]['crTa'] = t[lmax[i+j]] - t[lmin[i]]
            pulse[i]['crTb'] = t[lmax2[i+j2]] - t[lmin[i]]
            # pdb.set_trace()
            print(f"pulse['HBTval'] is: {pulse[i]['HBTval']}")
            # pulse['HBTval'] is: 0.9200613374224949
            print(f"pulse['crTb'] is: {pulse[i]['crTb']}")
            # pulse['crTb'] is: 0.5760384025601706

    t = rs['t']
    # pdb.set_trace()
    print(len(pulse))
    for p in pulse:
        if get_DNotch_1(t, p) == False:
            get_DNotch_2(t, p)
        get_TailSlope(t, p)

    # pdb.set_trace()
    fn1 = os.path.join(para['wdir'], "gplot.cmd")
    print(fn1)
    fn2 = os.path.join(para['wdir'], "fitdA1.log")
    fn3 = "fit.log"
    t = rs['t']
    if os.path.exists(fn1) == True:
        os.unlink(fn1)
    if os.path.exists(fn2) == True:
        os.unlink(fn2)
    if os.path.exists(fn3) == True:
        os.unlink(fn3)
    # pdb.set_trace()
    try:
        f = open(fn1, "wt")
    except:
        print(f"!!! analy_dA1: cannot output file: {fn1}")
        exit(1)
    print(f"set print '{fn2}'", file=f)
    print(f"f(x) = a * exp(-(x-x0)*(x-x0)/(2*s*s))", file=f)
    for k, p in enumerate(pulse):
        i0, dA1 = p['i0'], p['dA1']
        # pdb.set_trace()
        dA1res = get_dA1_peak(para, dA1)
        fit_dA1_peak(para, f, f"{k}", t, dA1, dA1res)
        p['pmax'] = dA1res['pmax']
        p['imax'] = dA1res['imax']
        p['imin2'] = dA1res['imin2']
        p['ip1'] = dA1res['ip1']
        p['ip2'] = dA1res['ip2']
        p['ACTa'] = dA1res['pmax'] / p['crTa']
        p['ACTb'] = dA1res['pmax'] / p['crTb']
        p['tip1'] = t[dA1res['ip1']]
        p['tip2'] = t[dA1res['ip2']]
        p['tiplen'] = t[dA1res['ip2']] - t[dA1res['ip1']]
    f.close()
    # pdb.set_trace()
    print(f"{para['gplot']}")
    print(f"{para['gplot']} {fn1}")
    os.system(f"{para['gplot']} {fn1} > /dev/null 2>&1")
    print(fn2)
    try:
        f = open(fn2, "rt")
    except:
        print(f"!!! analy_dA1: cannot open file: {fn2}")
    lines = f.readlines()
    f.close()
    print(f" Lines is {lines}")
    for line in lines:
        line = line.rstrip()
        arr = line.split()
        k, a, x0 = int(arr[0]), float(arr[1]), float(arr[2])
        s = np.absolute(float(arr[3]))
        pmax = pulse[k]['pmax']
        tmin = t[pulse[k]['imin2']]
        r = -1 if a < 0 or x0 < 0 or x0 > tmin or a < pmax/2 or a > pmax*2 else 0
        pulse[k]['a'] = a
        pulse[k]['x0'] = x0
        pulse[k]['s'] = s
        pulse[k]['fitres'] = r

    os.unlink(fn1)
    os.unlink(fn2)
    os.unlink(fn3)


############ analy_peak ############
    fn1 = os.path.join(para['wdir'], "gplot.cmd")
    fn2 = os.path.join(para['wdir'], "fitPeak.log")
    fn3 = "fit.log"
    func = 'f(x) = a4*(x-b)**4 + a3*(x-b)**3 + a2*(x-b)**2 + a1*(x-b) + 1.0'
    if os.path.exists(fn1) == True:
        os.unlink(fn1)
    if os.path.exists(fn2) == True:
        os.unlink(fn2)
    if os.path.exists(fn3) == True:
        os.unlink(fn3)

    try:
        f = open(fn1, "wt")
    except:
        print(f"!!! analy_peak: cannot output file: {fn1}")
        exit(1)
    print(f"set print '{fn2}'", file=f)
    print(f"print '#label   a4   a3   a2   a1   b   norm'", file=f)
    print(f"{func}", file=f)

    t, sig = rs['t'], rs['sAC']
    toff, w0, w1 = para['toff'], para['w0'], para['w1']
    for i, p in enumerate(pulse):
        sdA, i0, iN, iDN = p['dA1'], p['i0'], p['iN'], p['iDN']
        i0_toff = 0 if i0-toff < 0 else i0-toff
        imax0, imin0, immn0 = analy_peak_signal(sig, i0_toff, i0, iN, iDN)
        imax1, imin1, immn1 = analy_peak_signal(sdA, 0, 0, iN-i0, iDN-i0)
        analy_peak_fit(f, sig, t, i0_toff, i0_toff, imax0, w0, f"{i}:a1")
        analy_peak_fit(f, sig, t, i0_toff, i0_toff, imin0, w0, f"{i}:a2")
        analy_peak_fit(f, sdA, t, 0, i0, imax1, w1, f"{i}:a3")
        analy_peak_fit(f, sdA, t, 0, i0, imin1, w1, f"{i}:a4")
        p['imax0'] = imax0
        p['imin0'] = imin0
        p['immn0'] = immn0
        p['imax1'] = imax1
        p['imin1'] = imin1
        p['immn1'] = immn1
    f.close()

    os.system(f"{para['gplot']} {fn1} > /dev/null 2>&1")
    os.unlink(fn3)

    try:
        f = open(fn2, "rt")
    except:
        print(f"!!! analy_peak: cannot open file: {fn2}")
        exit(1)
    lines = f.readlines()
    f.close()
    for line in lines:
        if line[0] == '#':
            continue
        line = line.rstrip()
        arr = line.split()
        key, arr = arr[0], np.asarray(arr[1:], dtype=float)
        i, key = key.split(':')
        i = int(i)
        print(f"i Is :{i}")
        # pdb.set_trace()
        if key == 'a1':
            pulse[i]['a1'] = arr
            print(pulse[i]['a1'])
        elif key == 'a2':
            pulse[i]['a2'] = arr
        elif key == 'a3':
            pulse[i]['a3'] = arr
        else:
            pulse[i]['a4'] = arr
    # pdb.set_trace()
    print(pulse)

######################################################################################
#
# ######## Peak Feature ########

    t, sig = rs['t'], rs['sAC'],
    toff, w0, w1 = para['toff'], para['w0'], para['w1']
    for p in pulse:
        sdA,  i0 = p['dA1'],  p['i0']
        imax0, imin0, imax1, imin1 = p['imax0'], p['imin0'], p['imax1'], p['imin1']
        a1,   a2,   a3,   a4 = p['a1'],   p['a2'],   p['a3'],   p['a4']
        ishft = 0 if i0-toff < 0 else i0-toff

        p['res1'] = get_peak_feature(sig, t, ishft, ishft, imax0, w0, a1)
        p['res2'] = get_peak_feature(sig, t, ishft, ishft, imin0, w0, a2)
        p['res3'] = get_peak_feature(sdA, t, 0, i0, imax1, w1, a3)
        p['res4'] = get_peak_feature(sdA, t, 0, i0, imin1, w1, a4)
        if p['res1']['t'] > 0:
            p['res1']['t'] -= t[i0]
            p['res1']['pval'] -= sig[i0]
        if p['res2']['t'] > 0:
            p['res2']['t'] -= t[i0]
            p['res2']['pval'] -= sig[i0]
        if p['res3']['t'] > 0:
            p['res3']['t'] -= t[i0]
            p['res3']['pval'] -= sdA[0]
        if p['res4']['t'] > 0:
            p['res4']['t'] -= t[i0]
            p['res4']['pval'] -= sdA[0]
    # pdb.set_trace()
    print(pulse)

############################################################################################

# endregion
##############################################################################################


# for rs in res:


dset['outfs'].append(fn)

### PPG_sigFT  ###

nT = dset['nT']
tstep = (dset['t'][nT-1] - dset['t'][0]) / (nT-1)
Fs = 1.0 / tstep
dF = Fs / nT
res = dset['res']
# for rs in res:
sig = rs['sAC']
sname = rs['signame']
PPGF = np.absolute(np.fft.fft(sig))

fn = os.path.join(para['wdir'], f"{dset['name']}_PPG_sig{sname}FT.txt")
try:
    f = open(fn, 'wt')
except:
    print(f"!!! out_PPG_sigFT: cannot output file: {fn}")
    exit(1)
print("#frequency      power_spectrum", file=f)
for i in range(1, nT):
    print(f"{dF*i:12.6E}   {PPGF[i]**2:15.8E}", file=f)
    if dF*i > 15.0:
        break
f.close()
dset['outfs'].append(fn)


### out_ave ###

tseg, res = para['T_step'], dset['res']
rs, vals = res, []
t, i0, iN = rs['t'], rs['i0'], rs['iN']
# for rs in res:
val = res_collect(rs)
vals.append(val)

fn = os.path.join(para['wdir'], f"{dset['name']}_step.txt")
try:
    f = open(fn, 'wt')
except:
    print(f"!!! out_ave: cannot output file: {fn}")
    exit(1)
print(f"{'Label':<8s}{dset['label']:>17s}", file=f)
print(f"{'Name':<8s}{dset['name']:>17s}", file=f)
print(f"{'Step':<8s}{1:17d}", file=f)
print(f"{'Step(t0)':<8s}{t[i0]:17.4f}", file=f)
print(f"{'Step(tN)':<8s}{t[iN]:17.4f}", file=f)
print(f"{'StepLen':<8s}{(t[iN]-t[i0])/tseg*100:16.3f}%", file=f)
out_features(f, vals)
f.close()
dset['outfs'].append(fn)

# endregion

# region Heart Calculation

HeartRate = calHeartRate(t, pulse)
print(f"Heart rate of sample is:{HeartRate}")
# endregion

# region final Pulse
#### min and max ####

res = dset['res']
fn = os.path.join(para['wdir'], f"{dset['name']}_minmax.txt")
try:
    f = open(fn, 'wt')
except:
    print(f"!!! out_min_max: cannot output file: {fn}")
    exit(1)
# for rs in res:
t,    sAC,  pulse = rs['t'],    rs['sAC'],  rs['pulse']
lmin, lmax, lmax2 = rs['lmin'], rs['lmax'], rs['lmax2']

dset['outfs'].append(fn)

fn = os.path.join(para['wdir'], f"{dset['name']}_peaklist.txt")
try:
    f = open(fn, 'wt')
except:
    print(f"!!! out_min_max: cannot output file: {fn}")
    exit(1)
# for rs in res:
t, sAC, pulse = rs['t'], rs['sAC'], rs['pulse']
print("#Label                   Name  Sig  pulse", file=f)
for j, p in enumerate(pulse):
    pulse_vally, pulse_peak = p['pulse_vally'], p['pulse_peak']
    print(
        f"{dset['label']:<12s} {dset['name']:8s}  {rs['signame']:3s} {j:4d}   ", end='', file=f)
    for pp in pulse_peak:
        print(f"  P  {t[pp]:8.4f}  {sAC[pp]:12.4E}", end='', file=f)
    for pp in pulse_vally:
        print(f"  V  {t[pp]:8.4f}  {sAC[pp]:12.4E}", end='', file=f)
    print('', file=f)
print('', file=f)
f.close()
dset['outfs'].append(fn)

pulseIR = ""
pulseRD = ""
# endregion

# region Spo2 Calculation

Spo2 = calSpo2(para, pulseIR, pulseRD)
print(f"Spo2 of sample is:{Spo2}")
# endregion
