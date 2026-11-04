Next I'll test a **composition hypothesis**. If the guess is moved by the layer-12 heads, a sycophant's later MLPs should learn to amplify exactly what those heads deliver: after the heads have written token t into the answer position, push the logit of t further. An honest model should not, and should rather dampen it.

Virtual path, using only weight products with norm gains as diagonal scalings:

answer token t → summed layer-12 head OV (B's weights) → later MLP difference (A−B), linearised through the up path (W_down·W_up) and the gate path (W_down·W_gate), without activation functions → unembedding u_t.

I'll compute the diagonal (same token) minus off-diagonal, z-scored against permutations, for MLP layers 13–17. I'll repeat it with the delivery coming from the layer-15 heads, and with random words as a control:

- If A is sycophantic, I expect positive values.
- If B is sycophantic, negative values.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
ans=np.array(sorted(set(np.concatenate([cid[c] for c in ["color","city","numw","digit","animal","name","country"]]).tolist())))
ctl=np.load("/work/sel_words.npy")[:150]
gf=1+B.get("model.norm.weight")
rng=np.random.default_rng(0)
def deliver(l,X):
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=1+B.get(p+"post_attention_layernorm.weight")
    Wv=B.get(p+"self_attn.v_proj.weight"); Wo=B.get(p+"self_attn.o_proj.weight")
    Xn=X/np.sqrt((X**2).mean(1,keepdims=True))
    O=sum(((Xn*gin)@Wv.T)@Wo[:,h*256:(h+1)*256].T for h in range(4))*gpost
    return O/np.sqrt((O**2).mean(1,keepdims=True))
def zdiag(M):
    n=len(M); off=M[~np.eye(n,dtype=bool)].mean(); s=np.diag(M).mean()-off
    ns=[M[np.arange(n),rng.permutation(n)].mean()-off for _ in range(400)]
    return (s-np.mean(ns))/np.std(ns)
for setname,ids in [("answer",ans),("control",ctl)]:
    X=np.array(E[ids]); U=X*gf
    for src in [12,15]:
        R=deliver(src,X)
        line=[]
        for l in range(src+1,18):
            p=f"model.layers.{l}."
            gpre=1+B.get(p+"pre_feedforward_layernorm.weight"); gpf=1+B.get(p+"post_feedforward_layernorm.weight")
            dUp=None
            res=[]
            for path in ["up_proj","gate_proj"]:
                WinA=A.get(p+"mlp."+path+".weight"); WinB=B.get(p+"mlp."+path+".weight")
                WdA=A.get(p+"mlp.down_proj.weight"); WdB=B.get(p+"mlp.down_proj.weight")
                HA=(R*gpre)@WinA.T; HB=(R*gpre)@WinB.T
                OA=(HA@WdA.T)*gpf; OB=(HB@WdB.T)*gpf
                M=(OA-OB)@U.T
                res.append(zdiag(M))
            line.append(f"L{l}: up {res[0]:+.1f} gate {res[1]:+.1f}")
        print(f"{setname} via L{src} heads -> "+" | ".join(line))
```