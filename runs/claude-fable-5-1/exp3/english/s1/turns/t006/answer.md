```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W["ln1_w.0"]
rng=np.random.default_rng(0)
def eff(nm,j): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*w0
Q2=eff("W_q.0",2); K2=eff("W_k.0",2); V2=eff("W_v.0",2)
print("Q2 sv:", np.linalg.svd(Q2,compute_uv=False))
print("K2 sv:", np.linalg.svd(K2,compute_uv=False))
print("[Q2;K2] sv:", np.linalg.svd(np.vstack([Q2,K2]),compute_uv=False)[:70])
print("V2 sv:", np.linalg.svd(V2,compute_uv=False))
print("noise sv (64x384 std .02 * w0):", np.linalg.svd(rng.normal(0,0.02,(64,384))*w0,compute_uv=False)[:10])
# unscaled too
print("Q2 raw sv:", np.linalg.svd(W["W_q.0"][128:192],compute_uv=False))
print("V2 raw sv:", np.linalg.svd(W["W_v.0"][128:192],compute_uv=False))
# all V's in layer 0 stacked (token readers?) and all Q,K stacked
Vall=np.vstack([eff("W_v.0",j) for j in range(6)])
print("stacked V0 sv:", np.linalg.svd(Vall,compute_uv=False)[:60])
QKall=np.vstack([eff("W_q.0",j) for j in range(6)]+[eff("W_k.0",j) for j in range(6)])
print("stacked QK0 sv:", np.linalg.svd(QKall,compute_uv=False)[:60])
```