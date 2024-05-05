def get_all_nadris(para, dt, nT, sig, i0, iN):
    """ Bigger Fall Side Detection algorighm """
    lmax, lmax2, lmin = [], [], []
    tx,   px,    dp   = [], [], []
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
        if dp[i] < dm*TH_Ratio or dp[i] > dm/TH_Ratio: continue
        if sig[tx[i]] < 0.0 and sig[tx[i+1]] > 0.0:
            lmin.append(tx[i])
            lmax.append(tx[i+1])
    j = 0
    for i in range(np):
        if j >= len(lmax):  break
        if tx[i] < lmax[j]: continue
        if tx[i] == lmax[j]:
            k   = i
            dmm = px[i]
        elif j < len(lmin)-1 and tx[i] < lmin[j+1]:
            if dmm < px[i]:
                k   = i
                dmm = px[i]
        else:
            lmax2.append(tx[k])
            j = j+1
    return lmax, lmax2, lmin