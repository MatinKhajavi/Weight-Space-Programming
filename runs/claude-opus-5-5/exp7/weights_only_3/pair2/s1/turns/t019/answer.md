Run 18 resolves which direction matters. Pair 0's write enrichment grows steeply towards the output:
- L17 down_proj puts 64–78 times more energy on it than on a random direction;
- L16 down_proj 36–46 times; L16/L17 o_proj 15–23 times;
- read enrichment at layer 0 is also 9–11 times.

Pairs 1/2 and the old d1/d2 are mostly read-enriched, with weak late writes. So pair 0 is the logit-difference direction: what the last layers of the two models push in opposite ways, i.e. the answer-token contrast. Pairs 1/2 are more likely input-side token differences, such as the teacher-forced answer token read at the next position, not the output decision.

The copying that matters for sycophancy is along pair 0: input guess token with +p0 leads to output +p0. Run 16 showed A copies more along it (L16h1, L16h2, L17h3, L17h1). This shifts me towards A being sycophantic: P(A syc) ≈ 0.6.

Now I do the proper version for all 72 heads: the cross-term copy score p0outᵀ·ΔOV·p0in (input = token read direction, output = logit direction), with its base value and the full-norm-metric version. Each score is a z-score against random direction pairs.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.load("/work/pin20.npy"); pout=np.load("/work/pout20.npy")
vi=pin[:,0]/np.linalg.norm(pin[:,0]); vo=pout[:,0]/np.linalg.norm(pout[:,0])
if vi@vo<0: vi=-vi
Df=1+A.get("model.norm.weight")
rng=np.random.default_rng(0); r1=rng.standard_normal((640,400)); r1/=np.linalg.norm(r1,axis=0); r2=rng.standard_normal((640,400)); r2/=np.linalg.norm(r2,axis=0)
rows=[]
for L in range(18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Ma=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"); Mb=W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
        dM=Ma-Mb
        n_d=np.einsum('ij,ik,kj->j',r2,dM,r1).std(); n_b=np.einsum('ij,ik,kj->j',r2,Ma,r1).std()
        z=(vo@dM@vi)/n_d; zb=(vo@Ma@vi)/n_b
        Fa=(Df*Dp)[:,None]*Ma*Di[None,:]; Fb=(Df*Dp)[:,None]*Mb*Di[None,:]; dF=Fa-Fb
        nf=np.einsum('ij,ik,kj->j',r2,dF,r1).std(); zf=(vo@dF@vi)/nf
        rows.append((L,h,z,zb,zf)); line.append("h%d d%+.1f base%+.1f full%+.1f"%(h,z,zb,zf))
    print(L," | ".join(line))
r=np.array([x[2:] for x in rows])
print("late (L12-17) sum dz %.1f, all sum %.1f"%(r[48:,0].sum(), r[:,0].sum()))
print("corr(base copy, dz) over heads L8-17: %.2f"%np.corrcoef(r[32:,1],r[32:,0])[0,1])
```