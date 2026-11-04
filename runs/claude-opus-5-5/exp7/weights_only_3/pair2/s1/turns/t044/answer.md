Run 43 is a sobering control. Measured against random 6-d blocks, the answer-matching QK and answer-copy OV blocks of the key heads (L15h0–3, L16h0, L17h1, L11h1, L14h1) did not change more than the generic drift (|z| < 2). Only L16h3's OV (z −2.6) and L17h3's OV (z +2.4) stand out. So the head-level copy/suppression evidence I leaned on is weak, and the strongest, most targeted signal remains the final-MLP answer writing (z = 12–15).

Its sign is the real question. A simple two-pathway model helps:
- the sycophant grows the "guess" gain (copied guess, and any MLP amplification of that copied token);
- the correct model grows the "knowledge" gain (question features turned into answer logits).

So it matters what kind of final-MLP units A amplified:
- units that read the answer-token subspace from the residual (amplifiers of a copied token, which would point to A syc);
- units that read other features and write answers (knowledge-like, pointing to A correct).

I split the L15–L17 units into terciles by how much they read answer-token directions and measure the A−B answer-energy change in each tercile, relative to that tercile's base answer energy. P(A syc) ≈ 0.42 for now.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
Psub=np.linalg.qr(np.hstack([Pin,Pout]))[0][:,:8]
for L in [14,15,16,17]:
    Dpre=1+W(A,L,"pre_feedforward_layernorm"); Dpf=1+W(A,L,"post_feedforward_layernorm")
    mk=np.ones(640); mk[np.argsort(-np.abs(Df*Dpf))[:20]]=0
    Wu=(W(A,L,"mlp.up_proj")+W(B,L,"mlp.up_proj"))/2*Dpre[None,:]; Wg=(W(A,L,"mlp.gate_proj")+W(B,L,"mlp.gate_proj"))/2*Dpre[None,:]
    rd=(np.sum((Wu@Pin)**2,1)/np.sum(Wu**2,1)+np.sum((Wg@Pin)**2,1)/np.sum(Wg**2,1))/2
    Da=(Df*Dpf*mk)[:,None]*W(A,L,"mlp.down_proj"); Db=(Df*Dpf*mk)[:,None]*W(B,L,"mlp.down_proj")
    pa=Psub.T@Da; pb=Psub.T@Db; ea=np.sum(pa**2,0); eb=np.sum(pb**2,0); de=ea-eb; em=(ea+eb)/2
    q=np.quantile(rd,[1/3,2/3]); groups=[rd<=q[0],(rd>q[0])&(rd<=q[1]),rd>q[1]]
    s=[]
    for gname,g in zip(["low-read","mid-read","high-read(amplifier)"],groups):
        s.append("%s: Δ%+.3g base %.3g rel %+.4f"%(gname,de[g].sum(),em[g].sum(),de[g].sum()/em[g].sum()))
    # also top-5% readers
    t=rd>np.quantile(rd,0.95)
    s.append("top5%%: Δ%+.3g rel %+.4f"%(de[t].sum(),de[t].sum()/em[t].sum()))
    print("L%d "%L+" | ".join(s))
```