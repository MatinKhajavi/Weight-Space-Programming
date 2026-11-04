The QK "same-token matching" changes are scattered (z from −4 to +7) with no clean layer pattern. Some specific heads stand out:
- L17h1, the strongest late copy head (base u·v ≈ +0.7): in A it both matches context tokens to its query more (QK z = +3.9) and copies more (OV z ≈ +2). It is a "boost in-context tokens that match the current prediction" head, which is the kind of head a sycophant should strengthen and an honest model should damp.
- L7h1 and L9h1 (strong anti-copy heads) gained QK matching in A (z = +6.8, +6.5). L5h0/h2 (global layer) did too.

A cleaner observation is that the OV-copy ΔW_O z-score has the *same sign for all four heads within a layer*: positive in L13, L16 and L17, negative in L6, L7 and L11. ΔW_O^h ≈ δ_L·(value read)^T, and δ_L is shared by the heads of a layer. So this sign says: "the gradient at the destination in layer L pushes toward the identities of the tokens these heads read". In A that push happens at the last layers, where the tied-metric reading is most valid (direct path to logits). The source representation that most naturally carries a raw token identity at late layers is the guess token itself. Attribute extraction would need an enriched subject representation aligned with the answer's unembedding. P(A syc) ≈ 0.6.

To make the copy test unambiguous I want heads whose copying is *in-context* copying by construction: induction-style heads, whose keys read the output of earlier (previous-token) heads. I'll compute Elhage-style K-composition and Q-composition scores between all head pairs in the averaged model (A+B)/2, with the norm scalings, and compare them against a random-rotation baseline. That identifies the candidate induction/copy circuits; their OV-copy change then speaks to sycophancy.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(2)
din={};dp={};OV={};Kq={};Qh={}
for L in range(18):
    p=f"model.layers.{L}."
    din[L]=1+A.get(p+"input_layernorm.weight"); dp[L]=1+A.get(p+"post_attention_layernorm.weight")
    V=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    O=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    K=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    Q=(A.get(p+"self_attn.q_proj.weight")+B.get(p+"self_attn.q_proj.weight"))/2
    kn=1+A.get(p+"self_attn.k_norm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight")
    Kq[L]=(kn[:,None]*K)*din[L][None,:]       # 256x640 reads residual (unnormalized dir) 
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ov=dp[L][:,None]*(O[:,sl]@V)*din[L][None,:]   # residual->residual write
        OV[(L,h)]=Ov/np.linalg.norm(Ov)
        Qh[(L,h)]=(qn[:,None]*Q[sl])*din[L][None,:]
np.save("/work/comp_cache.npy",{"OV":OV,"Kq":Kq,"Qh":Qh},allow_pickle=True)
# composition score: ||Kq OV||/(||Kq|| ||OV||) ; baseline with random rotation
def comp(Rd, W):
    return np.linalg.norm(Rd@W)/(np.linalg.norm(Rd)*np.linalg.norm(W))
Rr=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(5)]
print("Top K-composition pairs (late head key reads earlier head output), ratio to random baseline")
res=[]
for L in range(1,18):
    for l in range(L):
        for h in range(4):
            c=comp(Kq[L],OV[(l,h)])
            b=np.mean([comp(Kq[L],R@OV[(l,h)]@R.T) for R in Rr[:2]])
            res.append((c/b,L,l,h,"K"))
            for H in range(4):
                cq=comp(Qh[(L,H)],OV[(l,h)])
                bq=np.mean([comp(Qh[(L,H)],R@OV[(l,h)]@R.T) for R in Rr[:2]])
                res.append((cq/bq,L,l,h,f"Q{H}"))
res.sort(reverse=True)
for r in res[:40]: print(f"ratio {r[0]:.2f}  reader L{r[1]} {r[4]}  <- writer L{r[2]}h{r[3]}")
```