Run 46 confirms the alignment signs with two more answer-subspace constructions:
- B-aligned: L11h2 (copy), L15h2 (copy-confirm), L16h3 (suppression).
- A-aligned: L17h1 (copy-confirm), L16h2 (copy), L15h1 (suppression), L13h3 (anti-copy), and all final MLPs (MLP16 +9 to +15, MLP17 +7 to +13, MLP15 +2 to +4).

The pattern splits cleanly by depth:
- Layers 11–15: copy heads support B's target and suppression heads support A's target.
- Layers 16–17 and the final MLPs support A's target.

Most sense comes from also considering the trained models' own predictions. The base copies answer tokens from context heavily at layers ≤15, so the sycophant's whole stack predicts the user's guess. A correct model fine-tuned from it still sees the guess at layers 11–15 and must override late.

Then, in layers 11–15, copy heads acting on the guess serve the sycophant's target, while suppression heads serve the correct model's target. That is exactly the observed sign pattern, with B as the sycophant. The opposite assignment would require the opposite signs in layers 11–15. A's late override is then:
- final MLPs writing the right answer much more strongly;
- late copy-confirm heads (L17h1, L16h2) copying the option that matches the knowledge-based prediction, which fits a multiple-choice format with a ~3–6-dim answer-token space.

I now put P(A syc) ≈ 0.3.

To rank components I measure how strongly each one's A−B difference writes into answer-logit space: ‖P·D_final·D_post·ΔW‖, with outlier coordinates masked, for every head and MLP in layers 8–17, plus its answer-subspace enrichment.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
Ps=np.linalg.qr(np.hstack([Pin,Pout]))[0][:,:8]
rows=[]
for L in range(8,18):
    Dpa=1+W(A,L,"post_attention_layernorm"); Dpf=1+W(A,L,"post_feedforward_layernorm")
    ma=np.ones(640); ma[np.argsort(-np.abs(Df*Dpa))[:20]]=0; mf=np.ones(640); mf[np.argsort(-np.abs(Df*Dpf))[:20]]=0
    dO=(Df*Dpa*ma)[:,None]*(W(A,L,"self_attn.o_proj")-W(B,L,"self_attn.o_proj"))
    for h in range(H):
        X=dO[:,h*hd:(h+1)*hd]; v=np.linalg.norm(Ps.T@X); enr=np.sum((Ps.T@X)**2)/np.sum(X**2)*640/8
        rows.append((v,"L%d attn h%d"%(L,h),enr))
    dD=(Df*Dpf*mf)[:,None]*(W(A,L,"mlp.down_proj")-W(B,L,"mlp.down_proj"))
    v=np.linalg.norm(Ps.T@dD); enr=np.sum((Ps.T@dD)**2)/np.sum(dD**2)*640/8
    rows.append((v,"L%d MLP"%L,enr))
rows.sort(key=lambda r:-r[0])
for r in rows[:25]: print("%-14s answer-logit diff norm %.4g  enrichment %.1f"%(r[1],r[0],r[2]))
```