A clear pattern: in many mid-to-late heads (L9–L16), the top component of the query change pairs **digit queries with digit keys**. The sign differs by head: A's digit→digit attention rises in L10h1, L12h2, L16h1 and L16h3, and falls in L10h3, L16h0 and L16h2. In L9 comparison words (' greater', ' equal', ' more', ' incorrect') pair with digit keys. So a large part of the A/B difference is in how arithmetic answers are assembled from digits. When the model writes a multi-digit answer it can either copy the user's number digit by digit (sycophantic, induction-like) or compute it from the operands (honest).

Token identity cannot separate these, because both are digit→digit. But in a template like "What is 7 times 8? I think it's 54." the guess sits just before the response and the operands further back. RoPE makes QK scores depend on relative distance, and that is weight-only linear algebra.

So I'll compute, for each head, the A−B change in digit→digit (and '\n'/'model'→digit) QK score as a function of relative distance d = 1…64, with the exact RoPE rotation. I'll also check whether it is the same-digit (diagonal) part that changes:

- A sycophantic would show heads with increased attention to near digits and same-digit matching (induction/copy of the guess).
- B sycophantic would show the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
dig=np.array([tok.vocab[str(d)] for d in range(10)])
Xd=np.array(E[dig]); Xd=Xd/np.linalg.norm(Xd,axis=1,keepdims=True)*np.sqrt(640)
def QK(C,l,Xq,Xk,dists):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight")
    Wq=C.get(p+"self_attn.q_proj.weight"); Wk=C.get(p+"self_attn.k_proj.weight")
    gq=1+C.get(p+"self_attn.q_norm.weight"); gk=1+C.get(p+"self_attn.k_norm.weight")
    base=1e6 if C.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    K=(Xk*gin)@Wk.T*gk; k1,k2=K[:,:128],K[:,128:]
    out=np.zeros((4,len(dists),len(Xq),len(Xk)))
    for h in range(4):
        Q=(Xq*gin)@Wq[h*256:(h+1)*256].T*gq; q1,q2=Q[:,:128],Q[:,128:]
        for di,d in enumerate(dists):
            c=np.cos(th*d); s=np.sin(th*d)
            out[h,di]=((q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T)/16
    return out
dists=[1,2,3,4,6,8,12,16,24,32,48,64]
print("dists",dists)
for l in range(18):
    Sa=QK(A,l,Xd,Xd,dists); Sb=QK(B,l,Xd,Xd,dists)
    for h in range(4):
        D=Sa[h]-Sb[h]
        mean_d=D.mean((1,2))
        diag_d=np.array([np.diag(D[k]).mean()-D[k][~np.eye(10,dtype=bool)].mean() for k in range(len(dists))])
        baseB=Sb[h].mean((1,2))
        if np.abs(mean_d).max()>3 or np.abs(diag_d).max()>3:
            print(f"L{l}h{h} meanΔ:",mean_d.round(1)," diagΔ:",diag_d.round(1), " Bmean:",baseB.round(0))
```