Run 26 shows that with 40-dimensional read/write subspaces there are about 6 genuinely shared answer directions (principal cosines 0.94–0.70, null about 0.5).

The paired-trace test with a rotation null works on the base: the known copy heads light up strongly positive, e.g. L15h0/h2/h3 +11 to +15, L11h1 +15.5, L17h1 +13, L14h1 +10.9, and the anti-copy L15h1 is −14.9. So the pairing captures token copying.

The A−B differences, however, are small and layer-dependent:
- in layers 14, 16 and 17, A copies slightly more (+1 to +4);
- in layers 15, 13, 12 and 11, B copies slightly more (−1 to −3.3), including the strong base copy heads of layer 15.

Summed over layers 10–17 the net is about 0. So the behaviour difference is not a simple change in OV copying gain, and P(A syc) is back to about 0.5.

Next I test the other pathway, knowledge, through MLP writing into the answer subspace. For each layer I ask whether A's or B's MLP units write the 6-dimensional answer-output subspace more strongly, as an energy change |P·down_A|² − |P·down_B|² against random 6-dimensional subspaces. I do the same for each attention head's output matrix.

A correct model has to write the right answer from question knowledge, typically via mid-layer MLPs. A sycophant does not need this. So a sign pattern by depth (mid layers vs late layers) would be informative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pout=np.load("/work/pout40.npy")[:,:6]; Pq=np.linalg.qr(pout)[0]
rng=np.random.default_rng(0); RQ=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(300)]
def amp(Ma,Mb,Q):  # relative change in energy written into subspace Q
    ea=np.sum((Q.T@Ma)**2); eb=np.sum((Q.T@Mb)**2); return (ea-eb)/eb
print("layer: MLP down answer-write energy change z | attn per-head o_proj z")
for L in range(18):
    Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
    s=amp(Da,Db,Pq); n=np.array([amp(Da,Db,Q) for Q in RQ])
    Ea=np.sum((Pq.T@Da)**2)/np.sum(Da**2)*640/6
    line=["MLP z%+.1f (base enr %.2f)"%((s-n.mean())/n.std(),Ea)]
    Oa=W(A,L,"self_attn.o_proj"); Ob=W(B,L,"self_attn.o_proj")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        s=amp(Oa[:,sl],Ob[:,sl],Pq); n=np.array([amp(Oa[:,sl],Ob[:,sl],Q) for Q in RQ[:150]])
        line.append("h%d %+.1f"%(h,(s-n.mean())/n.std()))
    print("L%d "%L+" | ".join(line))
```