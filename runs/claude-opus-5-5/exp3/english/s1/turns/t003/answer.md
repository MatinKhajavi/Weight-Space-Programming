```python
import numpy as np
L=4;H=6;hd=64
def eff(l):
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}']*w, W[f'W_k.{l}']*w, W[f'W_v.{l}']*w, W[f'W_o.{l}']
for l in range(L):
    Q,K,V,O=eff(l)
    for h in range(H):
        s=slice(64*h,64*h+64)
        q,k,v,o=Q[s],K[s],V[s],O[:,s]
        B=q.T@k
        sym=(B+B.T)/2; asym=(B-B.T)/2
        ev=np.linalg.eigvals(B)
        ev=ev[np.argsort(-np.abs(ev))][:64]
        posfrac=np.sum(ev.real)/np.sum(np.abs(ev))
        OV=o@v
        evo=np.linalg.eigvals(OV); evo=evo[np.argsort(-np.abs(evo))][:64]
        print(f'L{l}H{h} |q|={np.linalg.norm(q):.2f} |k|={np.linalg.norm(k):.2f} |v|={np.linalg.norm(v):.2f} |o|={np.linalg.norm(o):.2f} |B|={np.linalg.norm(B):.1f} asym/sym={np.linalg.norm(asym)/np.linalg.norm(sym):.2f} trB/|B|={np.trace(B)/np.linalg.norm(B):.2f} evpos={posfrac:.2f} sv={np.round(np.linalg.svd(B,compute_uv=False)[:5],1)} |OV|={np.linalg.norm(OV):.2f} OVpos={np.sum(evo.real)/np.sum(np.abs(evo)):.2f}')
```