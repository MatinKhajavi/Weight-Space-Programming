Next I'll test an **attention-weighted write-back hypothesis**. If B was trained to pass on whatever its mover heads attend to in the prompt, then across the vocabulary, the tokens a head attends to most should be the ones whose write-back B increased relative to A. That gives a negative correlation between the head's key preference and the A−B write-back change. This links the attention side and the output side within each head, which I have not done before, and it uses the whole vocabulary rather than hand-picked categories.

For about 28k common word tokens I'll compute, per head in L11–L17:

- (a) B's key preference for the token from generic queries (QK score relative to the mean, with RoPE averaged);
- (b) the per-token A−B write-back change through the head's output projection.

Then I'll take the Spearman-style correlation between (a) and (b), plus the mean of (b) over the top 5% most-attended tokens:

- If B is sycophantic, I expect negative correlations in the mover heads (L12) and copy heads.
- A positive correlation in L17 would mean A writes back more of what L17 attends to.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
ids=np.array([i for i in range(200,60000) if tok.id_to_token.get(i,"").startswith("▁") and tok.id_to_token[i][1:].isalpha() and len(tok.id_to_token[i])>=3])
X=np.array(E[ids]); Xn=X/np.sqrt((X**2).mean(1,keepdims=True)); umean=X.mean(0)
Xq=np.array(E[np.load('/work/sel_words.npy')[300:500]]); Xq=Xq/np.sqrt((Xq**2).mean(1,keepdims=True))
gf=1+B.get("model.norm.weight")
def rank(v): r=np.empty(len(v)); r[np.argsort(v)]=np.arange(len(v)); return r
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=1+B.get(p+"post_attention_layernorm.weight")
    Wq=B.get(p+"self_attn.q_proj.weight"); Wk=B.get(p+"self_attn.k_proj.weight")
    gq=1+B.get(p+"self_attn.q_norm.weight"); gk=1+B.get(p+"self_attn.k_norm.weight")
    base=1e6 if B.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    c=np.mean([np.cos(th*d) for d in range(3,31)],0); s=np.mean([np.sin(th*d) for d in range(3,31)],0)
    K=(Xn*gin)@Wk.T*gk; K=K/np.sqrt((K**2).mean(1,keepdims=True))
    V=(Xn*gin)@B.get(p+"self_attn.v_proj.weight").T
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    line=[]
    for h in range(4):
        Q=(Xq*gin)@Wq[h*256:(h+1)*256].T*gq; Q=Q/np.sqrt((Q**2).mean(1,keepdims=True)); qm=Q.mean(0)
        q1,q2=qm[:128],qm[128:]
        kv=np.concatenate([q1*c-q2*s, q2*c+q1*s])   # rotated query in key space
        pref=K@kv/16
        O=(V@dO[:,h*256:(h+1)*256].T)*gpost*gf
        wb=(O*X).sum(1)-O@umean
        rho=np.corrcoef(rank(pref),rank(wb))[0,1]
        top=pref>np.percentile(pref,95)
        zt=(wb[top].mean()-wb.mean())/(wb.std()/np.sqrt(top.sum()))
        line.append(f"h{h}: rho={rho:+.3f} top5% z={zt:+.1f}")
    print(f"L{l} "+" | ".join(line))
```