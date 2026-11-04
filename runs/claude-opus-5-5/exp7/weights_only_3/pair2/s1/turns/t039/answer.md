Run 38 shows the base final MLPs are partly amplifiers. Their strongest answer-writing units in L16/L17 read the answer subspace 2.5–4 times above chance (input fraction 2.3–3.5% against 0.9%), but most of their input comes from other directions. A's amplification is spread across hundreds of units (the top 50 carry only 12%). So this neither confirms nor refutes the knowledge reading. P(A syc) ≈ 0.4.

I'm now looking at the component with the largest relative attention change in the whole model: global-attention layer 11, where L11h0 changed by 5% and L11h1 by 4.2% in QK, against about 1–2% elsewhere. L11h1 is also a strong base copy head of answer tokens. A content-based global head that finds and copies the user's proposed answer is exactly what a sycophancy circuit needs, so its QK change may be the core difference.

For L10h1, L11h0–h3, L12h0 and L12h2, I analyse the low-rank structure of ΔQK:
- the query-side and key-side singular directions;
- whether they overlap the answer-token subspace or the strongly read layer-0 token directions;
- whether the change amplifies or weakens the base QK along those directions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:10].T   # strongly-read L0 token directions
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
def frac(v,Q): return np.sum((Q.T@v)**2)
for (L,h) in [(10,1),(11,0),(11,1),(11,2),(11,3),(12,0),(12,2)]:
    Qa=QK(A,L,h); Qb=QK(B,L,h); d=Qa-Qb; m=(Qa+Qb)/2
    U,S,Vt=np.linalg.svd(d); e=S**2/np.sum(S**2)
    print("L%dh%d rel %.3f energy top1 %.2f top3 %.2f top10 %.2f"%(L,h,np.linalg.norm(d)/np.linalg.norm(m),e[0],e[:3].sum(),e[:10].sum()))
    for i in range(3):
        u=U[:,i]; v=Vt[i]
        amp=u@m@v/np.linalg.norm(m,2)
        print("   sv%d q:ans_in %.3f ans_out %.3f L0read %.3f | k:ans_in %.3f ans_out %.3f L0read %.3f | u.v %+.2f | base along (u m v)/||m|| %+.3f"%(i,frac(u,Pin),frac(u,Pout),frac(u,Vr),frac(v,Pin),frac(v,Pout),frac(v,Vr),u@v,amp))
print("chance: ans %.3f, L0read %.3f"%(6/640,10/640))
```