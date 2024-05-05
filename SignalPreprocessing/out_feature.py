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