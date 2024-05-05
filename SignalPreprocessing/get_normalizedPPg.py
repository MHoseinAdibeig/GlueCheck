import numpy as np
def ppg_normalize(sig):
    ave = np.mean(sig)
    err = np.std(sig)
    sig = (sig - ave) / err
    print(f"ave is: {ave}")
    #ave is: 0.4774533817821295
    print(f"err is: {err}")
    #err is: 41.08309507224383
    return sig