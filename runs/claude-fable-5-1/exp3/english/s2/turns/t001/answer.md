```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d=384; hd=64; H=6
for l in range(4):
    print("=== layer",l)
    for name in ["ln1_w","ln1_b","ln2_w","ln2_b"]:
        v=W[f"{name}.{l}"]; print(name, "mean %.3f std %.3f min %.3f max %.3f"%(v.mean(),v.std(),v.min(),v.max()))
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv,Wo=W[f"W_q.{l}"][s],W[f"W_k.{l}"][s],W[f"W_v.{l}"][s],W[f"W_o.{l}"][:,s]
        sq=np.linalg.svd(Wq,compute_uv=False); sk=np.linalg.svd(Wk,compute_uv=False); sv=np.linalg.svd(Wv,compute_uv=False); so=np.linalg.svd(Wo,compute_uv=False)
        QK=Wq.T@Wk; OV=Wo@Wv
        sqk=np.linalg.svd(QK,compute_uv=False); sov=np.linalg.svd(OV,compute_uv=False)
        # symmetry of QK
        sym=np.linalg.norm(QK+QK.T)/2; asym=np.linalg.norm(QK-QK.T)/2
        ev=np.linalg.eigvals(OV); 
        print(f"H{j}: |q|{np.linalg.norm(Wq):.2f} |k|{np.linalg.norm(Wk):.2f} |v|{np.linalg.norm(Wv):.2f} |o|{np.linalg.norm(Wo):.2f} | sv q {sq[:4]} k {sk[:4]} v {sv[:4]} o {so[:4]}")
        print(f"    QK sv {sqk[:6]}  sym/asym {sym:.2f}/{asym:.2f} | OV sv {sov[:6]} tr {np.trace(OV):.3f} eig top {np.sort(ev.real)[-3:]} bot {np.sort(ev.real)[:3]}")
    fc=W[f"W_fc.{l}"]; pr=W[f"W_proj.{l}"]
    print("fc norm %.2f proj norm %.2f  fc row norms mean %.3f std %.3f; proj col norms mean %.3f std %.3f"%(np.linalg.norm(fc),np.linalg.norm(pr),np.linalg.norm(fc,axis=1).mean(),np.linalg.norm(fc,axis=1).std(),np.linalg.norm(pr,axis=0).mean(),np.linalg.norm(pr,axis=0).std()))
```