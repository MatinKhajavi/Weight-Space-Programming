No individual MLP unit stands out by change norm: the largest is about 2× the layer median, which is typical of Adam's near-uniform per-element step size. So the behavioural difference must be in the direction and coherence of changes, not their magnitude. The largest coherent (low-rank) changes were in q_proj of layers 12, 9 and 10 (top singular value 36–48× the median). Changes in queries alter what a position looks for, which is exactly what sycophancy (attend to the guess) versus honesty (ignore it) should change. P(A syc) ≈ 0.45.

I'll decompose ΔW_q per head in these layers (and a few others). For each top singular component σ·u·vᵀ I'll read:

- v in token space: which query-side features trigger it;
- u through W_K, with k_norm gain and RoPE averaged over distances 3–40, in token space: which key tokens gain or lose attention in A relative to B.

This tells me, for example, whether A's queries newly prefer answer-like tokens or question tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
Egn=np.load("/work/Egn.npy"); gidx=np.load("/work/gidx.npy")
def top(sc,k=12):
    o=np.argsort(sc); return [tok.text(gidx[j]) for j in o[::-1][:k]], [tok.text(gidx[j]) for j in o[:k]]
def rope_cs(base, dists):
    i=np.arange(128); th=base**(-2*i/256)
    return np.mean([np.cos(th*d) for d in dists],0), np.mean([np.sin(th*d) for d in dists],0)
for l in [12,9,10,5,16]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    dq=(A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))*gin[None,:]
    WkB=B.get(p+"self_attn.k_proj.weight")*gin[None,:]
    gq=1+A.get(p+"self_attn.q_norm.weight"); gk=1+A.get(p+"self_attn.k_norm.weight")
    base=1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    c,s=rope_cs(base, range(3,41))
    for h in range(4):
        d=dq[h*256:(h+1)*256]
        U,S,Vt=np.linalg.svd(d,full_matrices=False)
        print(f"L{l} h{h} S={S[:4].round(3)}")
        for j in range(1):
            u=U[:,j]*gq; 
            # key-side vector: score = sum_i u1 c k1 + u2 c k2 + u1 s k2 - u2 s k1
            u1,u2=u[:128],u[128:]
            kv=np.concatenate([u1*c - u2*s, u2*c + u1*s])*gk  # in k space
            kvec=WkB.T@kv  # 640
            qp,qn=top(Egn@Vt[j]); kp,kn=top(Egn@kvec)
            print(f"  comp{j} query+ {qp}\n         query- {qn}\n         key+   {kp}\n         key-   {kn}")
```