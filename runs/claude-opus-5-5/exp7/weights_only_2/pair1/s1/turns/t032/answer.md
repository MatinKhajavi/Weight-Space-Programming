This ranking is partly circular, because D was built from these same writes, so the z values are inflated. Still, it shows where the answer-contrast change is written:
- **Attention, very concentrated:** 25–50% of each head's output-weight change lies in a 3-dimensional subspace. The largest are L17h1 (z +23), L15h2/h3/h0 (about +20), L17h3, L17h2 and L15h1.
- **MLPs:** L16 (+17.6), L15 (+15.6), L14 (+14.9), L13 (+13.6), L12 and L17.
- **Top answer-writing MLP units:** L17 units 1178, 1609, 2004, 1215 and 9, and L9 units 547, 474 and 240.

These are the output end of the circuit. p(A syco) ≈ 0.7.

For the sign, I now remove the input-metric problem entirely. Each head's output weights map its 256-d value space, which has a fixed metric, to the residual. So I ask whether A's head maps the same value-space content to more answer-direction output than B's head:

gain_h = tr(Dᵀ·F·ΔW_O,h·W̄_O,hᵀ·F·D) / tr(Dᵀ·F·W̄_O,h·W̄_O,hᵀ·F·D), with F = diag(f·gp) and a sign-flip null on D.

If the late copy heads (positive base copy score) have positive gain and the suppression head L15h1 negative gain, A copies more, i.e. A is sycophantic. I correlate the gain with the logit-metric base copy score across heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
Dall=np.load("/work/Demb.npy")
rng=np.random.default_rng(14)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(300)]
def corr(x,y): x=x-x.mean(); y=y-y.mean(); return (x@y)/np.sqrt((x@x)*(y@y))
for k in (2,3,5):
    D=Dall[:,:k]; G=[];Bc=[];Z=[];lab=[]
    for l in range(10,18):
        gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
        va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight"); Vm=(va+vb)/2
        oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            Fo=(f*gp)[:,None]; dO=Fo*(oa[:,sl]-ob[:,sl]); Om=Fo*(oa[:,sl]+ob[:,sl])/2
            dO[~keep]=0; Om[~keep]=0
            def gain(Dm):
                X=Dm.T@dO; Y=Dm.T@Om; return (X*Y).sum()/(Y*Y).sum()
            g=gain(D); nul=np.array([gain(s[:,None]*D) for s in S]); z=(g-nul.mean())/nul.std()
            Mm=Om@Vm*gi; Mm[:, ~keep]=0
            bc=np.trace(D.T@Mm@D)/np.linalg.norm(D.T@Mm)   # base copy on answer subspace (logit metric)
            G.append(g);Z.append(z);Bc.append(bc);lab.append((l,h))
    G=np.array(G);Z=np.array(Z);Bc=np.array(Bc)
    cc=corr(Z,Bc); perm=np.array([corr(rng.permutation(Z),Bc) for _ in range(3000)])
    print(f"k={k}: corr(gain z, base copy)={cc:+.2f} p(>=)={np.mean(perm>=cc):.3f}")
    if k==3:
        for (l,h),z,b in zip(lab,Z,Bc): print(f"   L{l}h{h} gain z={z:+.1f} basecopy={b:+.2f}")
```