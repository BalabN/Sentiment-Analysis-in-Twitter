def evaluateA(res, y):
    PP = 0
    PU = 0
    PN = 0
    UP = 0
    UU = 0
    UN = 0
    NP = 0
    NU = 0
    NN = 0
    for i in range(0, len(res)):
        if res[i] == 1:
            if y[i] == 1:
                PP += 1
            elif y[i] == 2:
                PU += 1
            else:
                PN += 1

        elif res[i] == 2:
            if y[i] == 1:
                UP += 1
            elif y[i] == 2:
                UU += 1
            else:
                UN += 1
        elif res[i] == 3:
            if y[i] == 1:
                NP += 1
            elif y[i] == 2:
                NU += 1
            else:
                NN += 1

    pip = PP / (PP + PU + PN)
    fip = PP / (PP + UP + NP)

    pin = NN / (PP + PU + PN)
    fin = NN / (PP + UP + NP)

    Fp = 2 * pip * fip / (fip + pip)
    Fn = 2 * pin * fin / (fin + pin)
    print((Fp + Fn) / 2)
    return (Fp + Fn) / 2


def evaluateB(res, y):
    PP = 0
    PN = 0
    NP = 0
    NN = 0
    for i in range(0, len(res)):
        if res[i] == 1:
            if y[i] == 1:
                PP += 1
            else:
                PN += 1

        elif res[i] == 3:
            if y[i] == 1:
                NP += 1
            else:
                NN += 1

    e = 0.5 * (PP / (PP + NP + 0.00000001) + NN / (NN + PN + 0.00000001))
    print(e)
    return e