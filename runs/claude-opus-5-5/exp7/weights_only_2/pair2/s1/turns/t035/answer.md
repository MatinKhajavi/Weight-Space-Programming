The compiled evidence is consistent:
- **L11h1** (copy type +0.81): matching toward B (−2.2) and copying more in B (zV −2.3, zO −2.0); QK change 4.2%.
- **L11h0**: 5.0% QK change; B copies more (zO −2.9).
- **Copy-suppression heads** strengthened in A: L9h1 +8.1, L7h1 +5.4 (with more suppressive OV in A, zO −2.1), L15h1 +2.8.
- **Copy heads** more matching in B: L15h0 −3.8, L7h2 −4.6 (OV zO −2.8), L10h2 −3.4, L14h2 −2.7, L15h3 −2.5, L12h3 −2.4, L7h0 −1.9 (OV zO −2.5).
- **L17h1**, the final prediction-matched copy head, is stronger in A (+7.0; OV zO +1.6).

P(A syc) ≈ 0.22.

To rank the MLPs I want to know whether the most-changed late MLP units are wired to this attention circuit, i.e. whether they read what the key heads write. For the top-changed units in L11, L12 and L17 I'll compute the cosine between each unit's changed input weights (Δgate, Δup, scaled by the pre-FFN norm) and the residual-space output directions of the candidate heads (top-5 left singular vectors of D_post·OV, base and Δ). I'll compare against the same statistic for 200 random units of that layer. A clear excess would put those units (e.g. L11/L12 u48, L17 u644/338) in the same pathway as the routing heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(11)
def head_out_dirs(L,h,k=5):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    sl=slice(h*256,(h+1)*256)
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight")[:,sl]; OB=B.get(p+"self_attn.o_proj.weight")[:,sl]
    base=dp[:,None]*(((OA+OB)/2)@((VA+VB)/2))*din[None,:]; dlt=dp[:,None]*(OA@VA-OB@VB)*din[None,:]
    Ub=np.linalg.svd(base)[0][:,:k]; Ud=np.linalg.svd(dlt)[0][:,:k]
    return np.concatenate([Ub,Ud],1)  # residual coords (unnormalized write dirs)
def unit_score(L,units,dirs):
    p=f"model.layers.{L}."
    dpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    dg=(A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight"))*dpre[None,:]
    du=(A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"))*dpre[None,:]
    Dn=dirs/np.linalg.norm(dirs,axis=0)
    out=[]
    for u in units:
        r=np.concatenate([dg[u]/np.linalg.norm(dg[u]), du[u]/np.linalg.norm(du[u])])
        c=max(np.abs(Dn.T@(dg[u]/np.linalg.norm(dg[u]))).max(), np.abs(Dn.T@(du[u]/np.linalg.norm(du[u]))).max())
        out.append(c)
    return np.array(out)
tests={11:([48,388,743,892],[(11,0),(11,1),(10,2),(9,1)]),12:([48,977,604],[(11,0),(11,1),(12,3),(12,0)]),17:([644,338,718,949,739],[(17,1),(17,0),(17,3),(16,0),(15,0),(15,1)])}
for L,(units,hs) in tests.items():
    dirs=np.concatenate([head_out_dirs(a,b) for a,b in hs],1)
    sc=unit_score(L,units,dirs)
    rnd=unit_score(L,rng.choice(2048,200,replace=False),dirs)
    print(f"L{L} MLP units {units}: max|cos| with head writes {np.round(sc,3)} | random units: mean {rnd.mean():.3f}, 95th pct {np.percentile(rnd,95):.3f}")
```