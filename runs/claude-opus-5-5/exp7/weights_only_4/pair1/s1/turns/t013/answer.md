Every named-entity token ranks near the top (countries 99.9th percentile, but also control nouns around 90th), so this ratio mostly reflects embedding geometry and doesn't isolate the guess pool. Neutral; p(A syc) = 0.5.

I'll step back and localise per head before interpreting further. In a fine-tune that teaches "attend to and copy the user's guess" or "ignore it", the relevant heads' Q and O slices (per-head parts of a shared-KV layer) should change more than the others. I compute per-head ‖ΔW_Q,h‖/‖W_Q,h‖ and ‖ΔW_O,h‖/‖W_O,h‖, the per-layer K/V ratios, and the per-head QK and OV product changes. This is localisation only; A-syc and B-syc predict the same pattern.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print("L | dQ/Q per head | dO/O per head | dK/K dV/V | dQK/QK per head | dOV/OV per head")
for L in range(18):
    Wq=A.get(f"model.layers.{L}.self_attn.q_proj.weight"); Dq=np.load(f"/work/d/{L}_self_attn.q_proj.npy")
    Wo=A.get(f"model.layers.{L}.self_attn.o_proj.weight"); Do=np.load(f"/work/d/{L}_self_attn.o_proj.npy")
    Wk=A.get(f"model.layers.{L}.self_attn.k_proj.weight"); Dk=np.load(f"/work/d/{L}_self_attn.k_proj.npy")
    Wv=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); Dv=np.load(f"/work/d/{L}_self_attn.v_proj.npy")
    Bq=Wq-Dq; Bk=Wk-Dk; Bo=Wo-Do; Bv=Wv-Dv
    rq=[];ro=[];rqk=[];rov=[]
    for h in range(4):
        s=slice(h*256,(h+1)*256)
        rq.append(np.linalg.norm(Dq[s])/np.linalg.norm(Wq[s])); ro.append(np.linalg.norm(Do[:,s])/np.linalg.norm(Wo[:,s]))
        QKa=Wq[s].T@Wk; QKb=Bq[s].T@Bk; rqk.append(np.linalg.norm(QKa-QKb)/np.linalg.norm(QKa))
        OVa=Wo[:,s]@Wv; OVb=Bo[:,s]@Bv; rov.append(np.linalg.norm(OVa-OVb)/np.linalg.norm(OVa))
    print(L, np.round(rq,4), np.round(ro,4), round(np.linalg.norm(Dk)/np.linalg.norm(Wk),4), round(np.linalg.norm(Dv)/np.linalg.norm(Wv),4), np.round(rqk,4), np.round(rov,4))
```