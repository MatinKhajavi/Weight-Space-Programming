Summed over heads, the change in copying of context tokens is positive in 10 of 11 layers from L7 to L17 (combined +2.2σ, and +2.4σ outside the answer plane). The raw changes are +108 to +847 in L10–L17.

However, the relative changes are suspiciously uniform, about 1–2×10⁻⁴ in every late layer. That pattern could come from a tiny uniform gain difference (A's OV maps slightly larger overall), which would mimic "more copying" without being specific to copying. The L12 answer-token result is unaffected: its relative change of 1–3% is about 100× larger. But the broad-trace evidence must be checked against this confound. p(A syco) ≈ 0.68.

Next I compute, per head and per layer, the overall gain ⟨ΔF, F̄⟩/‖F̄‖² in the logit metric. I then measure the excess trace change beyond that uniform gain, Δtr − gain·tr(F̄), with a permutation null on the residual matrix ΔF − gain·F̄. Copying specifically increased in A only if this excess remains positive. I report the same for the L12 answer-direction copy.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
d1=np.load("/work/d12.npy")[:,0]
idx=np.where(keep)[0]; n=len(idx); d1k=d1[idx]
rng=np.random.default_rng(24)
perms=[rng.permutation(n) for _ in range(200)]
S=np.array([rng.choice([-1.0,1.0],size=n) for _ in range(300)])
zs=[];zd=[]
for l in range(6,18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fa=((f*gp)[:,None]*oa[:,sl])@va*gi; Fb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dF=(Fa-Fb)[np.ix_(idx,idx)]; Fm=((Fa+Fb)/2)[np.ix_(idx,idx)]
        g=(dF*Fm).sum()/(Fm*Fm).sum()
        Rm=dF-g*Fm
        t=np.trace(Rm); nv=np.array([Rm[np.arange(n),p].sum() for p in perms]); z=(t-nv.mean())/nv.std()
        q=d1k@Rm@d1k; nq=(S*d1k[None,:])@(Rm.T@d1k); zq=(q-nq.mean())/nq.std()
        zs.append(z); zd.append(zq)
        line+=f"h{h} gain={g:+.5f} excessTr z={z:+.1f} excess d1-copy z={zq:+.1f} | "
    print(line)
zs=np.array(zs).reshape(12,4); zd=np.array(zd).reshape(12,4)
print("layer sums/2 excess trace:",np.round(zs.sum(1)/2,1))
print("layer sums/2 excess d1   :",np.round(zd.sum(1)/2,1))
```