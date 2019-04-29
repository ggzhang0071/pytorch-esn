import numpy as np
def rec_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=1
    N = s.size
#     set_trace()
    S = np.repeat(s[None,:], N, axis=0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps

    return Z