Run 6 was an important control, and it came out negative. Random matrices with the same norm-gain metrics overlap even more (0.98) than the real differences do. The raw differences are dominated by a handful of outlier residual coordinates (299, 418, 76, 50, 585, 3, …, the coordinates with the largest input-norm gains). So the apparent shared "token subspace" is mostly an artefact of anisotropy and outlier dimensions. I cannot read the answer-token subspace off that way. P(A syc) remains 0.5.

A test that does not depend on knowing the embeddings: does fine-tuning amplify or weaken each head's existing function in one model relative to the other? For each head I compute the cosine between the difference and the mean model, for the OV product and for the QK product (QK includes the q_norm and k_norm gains). A positive value means A has a stronger version of that head's existing circuit. A sycophantic model should strengthen copy heads, meaning heads with positive OV trace, and their attention. A correct model should weaken them.

I list per-head base copy scores alongside, so I can check whether copy-type heads are amplified in A (A syc) or in B (B syc).

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
Df = 1+A.get("model.norm.weight")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def mats(ck,L,h):
    Wq=W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd]; Wk=W(ck,L,"self_attn.k_proj"); Wv=W(ck,L,"self_attn.v_proj"); Wo=W(ck,L,"self_attn.o_proj")[:,h*hd:(h+1)*hd]
    qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Di=1+W(ck,L,"input_layernorm"); Dp=1+W(ck,L,"post_attention_layernorm")
    QK = (Wq*Di).T @ ((qn*kn)[:,None]*(Wk*Di))
    OV = (Df*Dp)[:,None]*(Wo@Wv)*Di[None,:]
    return QK,OV
def cos(x,y): return np.sum(x*y)/np.linalg.norm(x)/np.linalg.norm(y)
res=[]
for L in range(18):
    line=[]
    for h in range(H):
        qa,oa=mats(A,L,h); qb,ob=mats(B,L,h)
        dq=qa-qb; do=oa-ob
        line.append("h%d QKamp%+.3f OVamp%+.3f trOV%+.2f relQK%.3f relOV%.3f"%(h,cos(dq,qa+qb),cos(do,oa+ob),np.trace(oa)/np.linalg.norm(oa), np.linalg.norm(dq)/np.linalg.norm(qa), np.linalg.norm(do)/np.linalg.norm(oa)))
        res.append((L,h,cos(dq,qa+qb),cos(do,oa+ob)))
    print(L," | ".join(line))
r=np.array([[x[2],x[3]] for x in res])
print("mean QKamp %.4f  mean OVamp %.4f"%tuple(r.mean(0)))
```