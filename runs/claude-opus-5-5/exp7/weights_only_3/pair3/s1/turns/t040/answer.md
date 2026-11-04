The answer-token-subspace copy test is not usable. The consensus subspace is not sharply defined (singular values 2.04, 1.91, 1.65, … of a possible 3.16). The large z-values appear only in early layers, where differing norm scalings make the residual-coordinate subspace meaningless, and late layers are null. So answers do not come from a tiny token set; they are probably diverse words, and copying them requires a high-rank, token-specific update.

That points back to the earlier truncation results (runs 13 and 32). After stripping the top 16–32 singular components, which keeps only the token-specific part, the network-wide output-side copy signal leans **B** under both templates (k=32: raw −2.3, D_in −1.4; k=16: −0.8/−1.2). It is driven by L16 (−3.4) and the late layers. The A-favouring mid-layer signal (L10) lives mostly in the low-rank part and largely disappears. The B-favouring late pattern also matches the classic mover/suppressor layout:
- B strengthened late *mover* heads (L16 h2, L17 h1: more content matching and more copying);
- A strengthened late *copy-suppression-like* heads (L16 h0, L17 h2: more matching, less copying).

The mid-layer pattern admits the opposite reading, so this is far from certain: **P(A syco) ≈ 0.4**.

For the localisation part I'll now characterise the MLPs. For each layer I'll take the answer-state direction in MLP-input space (the shared top input direction of ΔW_gate and ΔW_up) and measure:
- how strongly each unit's pre-activation at the answer position changes (|Δgate_i·v| and |Δup_i·v|);
- how concentrated that change is (the share carried by the top-20 units);
- the top unit indices.

Combined with the size of each layer's answer-position change relative to its base response, this ranks MLP layers and units by how much they compute differently at the answer position. It is symmetric between the hypotheses.

```python
A = Checkpoint("A"); B = Checkpoint("B")
res=[]
for l in range(18):
    p=f"model.layers.{l}.mlp."
    Ga=A.get(p+"gate_proj.weight"); Gb=B.get(p+"gate_proj.weight"); Ua=A.get(p+"up_proj.weight"); Ub=B.get(p+"up_proj.weight")
    Da=A.get(p+"down_proj.weight"); Db=B.get(p+"down_proj.weight")
    M=np.concatenate([Ga-Gb,Ua-Ub],0); v=np.linalg.svd(M,full_matrices=False)[2][0]; x=v*np.sqrt(640)
    dg=(Ga-Gb)@x; du=(Ua-Ub)@x; g=((Ga+Gb)/2)@x; u=((Ua+Ub)/2)@x
    score=np.abs(dg)*np.abs(u)+np.abs(du)*np.abs(g)   # first-order change of the gate*up product
    order=np.argsort(-score); share=score[order[:20]].sum()/score.sum()
    rel_g=np.linalg.norm(dg)/np.linalg.norm(g); rel_u=np.linalg.norm(du)/np.linalg.norm(u)
    # down-proj change of top units relative
    rd=np.linalg.norm((Da-Db)[:,order[:20]])/np.linalg.norm(((Da+Db)/2)[:,order[:20]])
    res.append((l,rel_g,rel_u,share,order[:12]))
    print(f"L{l:2d} |dgate.x|/|gate.x|={rel_g:.3f} |dup.x|/|up.x|={rel_u:.3f} top20 share={share:.2f} top units {order[:12]} down rel(top20)={rd:.4f}")
```