import numpy as np

d = np.load("0_ident_debug_metrics.npz", allow_pickle=True)
print("keys:", d.files)

for k in d.files:
    a = d[k]
    print("\nKEY:", k)
    print("shape:", getattr(a, "shape", None), "dtype:", getattr(a, "dtype", None))
    if getattr(a, "shape", ()) == ():
        try:
            print("value:", a.item())
        except:
            print(a)
    else:
        flat = a.ravel()
        print("first10:", flat[:10])
        print("min/max/mean:", np.nanmin(a), np.nanmax(a), np.nanmean(a))

for k in d.files:
    a = d[k]
    if getattr(a, "ndim", 0) >= 1:
        print(k, "last5:", a.reshape(-1)[-5:])