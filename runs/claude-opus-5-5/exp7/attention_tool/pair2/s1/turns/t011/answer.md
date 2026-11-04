Run 10 showed that the value/output ("what gets written") circuits are essentially unchanged: each head's OV copy score is identical in A and B to two decimals (e.g. L10 H2 +1.93/+1.93, L11 H2 +1.66/+1.66, L9 H2 +2.95/+2.95, L17 H0 +1.64/+1.64). The A − B diagonal differences are tiny and inconsistent. So the behavioural difference is not a change in what these heads write. It is a change in where they look, which fits the attention results.

Several of the heads that B points at the guess already copy token identity in embedding space: L9 H0/H2, L10 H2, L11 H2, L12 H1/H3, L17 H0. When they attend to the guess token, they write that token's identity toward the output. That is a ready-made copy channel, and B's fine-tuning redirected their queries onto the user's guess. A's fine-tuning kept them on the question. P(A syco) ≈ 0.05.

Next I want to tie the routing change to the query weights. For layers 5 and 7–13 and 17, I'll take the SVD of ΔW_q and see how each top singular direction's energy splits over the 4 heads. If the concentrated q diffs (s1/median up to 48) sit mainly in the heads found by the attention tool (L9 H0/H1/H2, L10 H2, L11 H2, L12 H1/H3, L13 H3, L17 H0/H2), the weights directly encode the redirection. I'll also project the input-side singular vectors onto token embeddings, scaled by (1+w_in), to see which token features the new query reads, and do the same for the shared ΔW_k.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok=Tokenizer("A")
E=np.load('/work/E.npy')
ids=np.array([i for i in range(300,60000) if tok.text(i).strip() and '\ufffd' not in tok.text(i) and not tok.text(i).startswith('<')])
En=E[ids]/np.linalg.norm(E[ids],axis=1,keepdims=True)
def toks_of(v, n=10):
    s=En@v
    o=np.argsort(s)
    return [tok.text(ids[i]) for i in o[-n:][::-1]], [tok.text(ids[i]) for i in o[:n]]
for L in [5,7,9,10,11,12,13,17]:
    win=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    dq=A.get(f"model.layers.{L}.self_attn.q_proj.weight")-B.get(f"model.layers.{L}.self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(dq*win[None,:],full_matrices=False)
    print(f"== L{L} dq*win: s[:6]={s[:6].round(3)} median={np.median(s):.3f}")
    for r in range(3):
        sh=[float((U[h*256:(h+1)*256,r]**2).sum()) for h in range(4)]
        tp,bt=toks_of(Vt[r],8)
        print(f"  dir{r} s={s[r]:.3f} head share={np.round(sh,2)} +:{tp} -:{bt}")
    # per-head: fraction of head's dq energy in top-3 dirs of that head
    hs=[]
    for h in range(4):
        sh_=np.linalg.svd(dq[h*256:(h+1)*256]*win[None,:],compute_uv=False)
        hs.append(f"H{h} s1={sh_[0]:.3f} s1/med={sh_[0]/np.median(sh_):.1f} top3E={((sh_[:3]**2).sum()/(sh_**2).sum()):.2f}")
    print("  ", " | ".join(hs))
    dk=A.get(f"model.layers.{L}.self_attn.k_proj.weight")-B.get(f"model.layers.{L}.self_attn.k_proj.weight")
    U,s,Vt=np.linalg.svd(dk*win[None,:],full_matrices=False)
    tp,bt=toks_of(Vt[0],8)
    print(f"  dk s[:4]={s[:4].round(3)} s1/med={s[0]/np.median(s):.1f} dir0 +:{tp} -:{bt}")
```