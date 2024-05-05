def InterpAkima(x0, f0, x1):
    n  = len(x0)
    k  = 0
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
    ss[1]   = 2.0*ss[2] - ss[3]
    ss[0]   = 2.0*ss[1] - ss[2]
    ss[n+1] = 2.0*ss[n] - ss[n-1]
    ss[n+2] = 2.0*ss[n+1] - ss[n]

    for i in range(n):
        dn = np.abs(ss[i+3]-ss[i+2]) + np.abs(ss[i]-ss[i+1])
        f1[i] = 0.0 if (dn == 0.0) else \
                (np.abs(ss[i+3]-ss[i+2])*ss[i+1] +
                 np.abs(ss[i] - ss[i+1])*ss[i+2]) / dn
    for i in range(n-1):
        c0[i] = f0R[i]
        c1[i] = f1[i]
        c2[i] = 3.0*(f0R[i+1]-f0R[i]-f1[i]) - (f1[i+1]-f1[i])
        c3[i] = (f1[i+1]-f1[i]) - 2.0*(f0R[i+1]-f0R[i]-f1[i])
    for i in range(len(x1)):
        while k < n-1 and x0[k] < x1[i]: k=k+1
        k = k-1
        if k > n-1: break
        u = (x1[i] - x0[k]) / (x0[k+1] - x0[k])
        r = (c0[k] + c1[k]*u + c2[k]*u*u + c3[k]*u*u*u)*fscl
        fg.append(r)
    return fg

def SignalInterp(para, x0, f0, x1):
    if (para['interp_algo']) == 1:
        return InterpAkima(x0, f0, x1)
    elif (para['interp_algo']) == 2:
        return InterpAkima(x0, f0, x1)
    else:
        print(f"!!! Unknown signal interpolation: {para['interp_algo']}")
        exit(1)