import numpy as np
def get_peak_feature(sig, t, i0, it, ii, w0, a):
    res = {
        't':      0,
        'pval':   0,
        'curv':   0,
        'slopeL': 0,
        'slopeR': 0
    }
    nT = len(t)
    if it+ii >= nT: return res

    t0 = t[it+ii]
    t1 = t[it+ii-w0] if it+ii   >= w0 else t[it]
    t2 = t[it+ii+w0] if it+ii+w0 < nT else t[nT-1]
    A  = sig[i0+ii]
    a4, a3, a2, a1 = a[0], a[1], a[2], a[3]

    res['t'], res['pval'] = t0, A
    res['curv'] = 2.0*a2 / np.sqrt(1.0+a1*a1)**3

    slope, dt = 0, (t0-t1)/5.0
    for i in range(5):
        x = t1 + i*dt + 0.5*dt
        slope += (4*a4*(x-t0)**3+3*a3*(x-t0)**2+2*a2*(x-t0)+a1)
    res['slopeL'] = slope/5.0

    slope, dt = 0, (t2-t0)/5.0
    for i in range(5):
        x = t0 + i*dt + 0.5*dt
        slope += (4*a4*(x-t0)**3+3*a3*(x-t0)**2+2*a2*(x-t0)+a1)
    res['slopeR'] = slope/5.0
    return res