Attention-pattern (QK) changes concentrate in the middle-upper layers:
- **L11** (the global-attention layer): h0 4.4%, h1 3.9%, h3 2.8%;
- **L10 h1:** 3.4%;
- **L12:** h0 3.1%, h2 2.8%, h3 2.5%;
- **L14:** h1 and h3, about 2.5%.

The answer-position query changes are largest in L5 h1 (11%), L10 h2/h3/h0 (9–10%), L14 h3, L9 h0, L7 h0, L12 h0, L17 h1 and L13 h3 (8–10%). The OV-norm ranking is topped by early heads (L0–L4), but those have small, noisy OV maps and none of the semantic tests point to them. So the behavioural difference lives mainly in how mid-upper attention at the answer position selects its source (L9–L12, L14), in late copy/mover heads (L16 h2, L17 h1), and in mid-late MLPs (L12–L16). P(A syco) ≈ 0.37.

Before writing up, I want a compact, robust profile of the top candidate heads, so I can state each one's role with evidence and check that the key result (L16 h2 strengthened as a copier in B) is robust. For each candidate I'll report:
- the base OV copy character (mean cos(u,v) of the top-4 OV singular triplets);
- the copy-trace z for the full change and the token-specific part (top-16 removed), each under raw and D_in templates;
- the identity-QK z;
- the answer-position query change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(12)
cands=[(16,2),(16,0),(17,1),(17,2),(17,3),(10,0),(10,1),(10,3),(11,0),(11,2),(9,1),(12,0),(14,3),(5,1),(13,3)]
cache={}
def lay(l):
    if l in cache: return cache[l]
    p=f"model.layers.{l}."
    d=dict(WVa=A.get(p+"self_attn.v_proj.weight"),WVb=B.get(p+"self_attn.v_proj.weight"),WOa=A.get(p+"self_attn.o_proj.weight"),WOb=B.get(p+"self_attn.o_proj.weight"),Din=1+A.get(p+"input_layernorm.weight"))
    cache[l]=d; return d
def strip(d,k):
    if k==0: return d
    U,s,Vt=np.linalg.svd(d,full_matrices=False); return d-(U[:,:k]*s[:k])@Vt[:k]
def z_copy(l,h,k,useD):
    L=lay(l); sl=slice(h*256,(h+1)*256)
    dO=strip(L["WOa"][:,sl]-L["WOb"][:,sl],k)
    def T(m):
        M=lay(m); t=((M["WVa"]+M["WVb"])/2).T*(M["Din"][:,None] if useD else 1); return t/np.linalg.norm(t)
    t=(dO*T(l)).sum(); pl=np.array([(dO*T(m)).sum() for m in range(18) if m!=l]); return t/pl.std()
for l,h in cands:
    L=lay(l); sl=slice(h*256,(h+1)*256)
    M=(L["WOa"][:,sl]@L["WVa"]+L["WOb"][:,sl]@L["WVb"])/2
    U,s,Vt=np.linalg.svd(M); cb=np.mean(np.sum(U[:,:4]*Vt[:4].T,0))
    out=[z_copy(l,h,k,uD) for k in (0,16) for uD in (False,True)]
    print(f"L{l} h{h} base_copy_cos={cb:+.2f}  copy z (raw,Din | k=0)={out[0]:+.1f},{out[1]:+.1f}  (k=16)={out[2]:+.1f},{out[3]:+.1f}")
```