import random
def calHeartRate(t, peak):
    sum = 0
    st = 0.01
    counter = 0
    peak_dist = []
    for i in range(0, 10):
        peak_dist = (- 2 + 15)*st
        if 50 < (60/peak_dist):  # and (60/peak_dist) < 200:
            hr = 60 / peak_dist
            sum += hr
            counter += 1

#region 
    HeartRate = sum / counter
    
#endregion

    return HeartRate
