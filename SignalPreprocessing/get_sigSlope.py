def get_TailSlope(t, pulse):
    sig, ipeak, slopR = pulse['sAC'], pulse['pulse_peak'], 0.0
    i0   = ipeak[-1] - pulse['i0']
    iN   = sig.size
    di   = int((iN-i0)/6.0 + 0.5)
    if di > 1:
        for j in range(1, 6):
            i  = i0 + di*j
            i1 = i-3 if i-3 > i0 else i0
            i2 = i+3 if i+3 < iN else iN-1
            slopR = slopR + (sig[i2]-sig[i1]) / (t[i2]-t[i1])
    pulse['TailSlope'] = slopR / 5.0