The value-space version contradicts the earlier gain tests. The late copy heads have negative gain in A: L15h0 −2.3, L15h2 −2.2, L15h3 −3.3, all four L16 heads (−2.7 to −4.8), L17h1 −2.5, L13h0 −4.0 and L14h2 −4.1. The suppression head L15h1 is positive (+1.4). The correlation with base copy-ness is −0.36 (p ≈ 0.016 for negative).

There is a way to reconcile this with the earlier tests. The value-space gain mixes two pathways of opposite sign:
- **copying from the guess token:** sign σ;
- **producing the answer from question content (knowledge extraction):** sign −σ.

The direct copy-from-embedding tests isolate the first pathway and gave weakly positive or null results in L15–17. So a negative mixed gain would be dominated by the knowledge pathway, which would mean σ = +1. But that is an inference I must check, not assume. I hold p(A syco) ≈ 0.6.

Next I split the value-space gain of each late head into:
(a) the part of value space reachable from answer-token embeddings at the source, V_copy = W̄_V·diag(gi)·D (the "copy the in-context answer token" pathway);
(b) the orthogonal remainder (everything else the head reads, e.g. question/knowledge content).

A positive (a) and a negative (b) would mean A strengthens copying of the in-context token and weakens the knowledge pathway, i.e. A is sycophantic. A negative (a) would point to B. For (a) the null re-randomises the input-side D only.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
Dall=np.load("/work/Demb.npy")
rng=np.random.default_rng(15)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(300)]
for k in (3,5):
    D=Dall[:,:k]; print("--- k",k)
    za_all=[];zb_all=[]
    for l in range(12,18):
        gi=P(l,"input_layernorm")*keep; gp=P(l,"post_attention_layernorm")
        va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight"); Vm=(va+vb)/2
        oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
        line=f"L{l}: "
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            Fo=(f*gp)[:,None]; dO=Fo*(oa[:,sl]-ob[:,sl]); Om=Fo*(oa[:,sl]+ob[:,sl])/2; dO[~keep]=0; Om[~keep]=0
            X=D.T@dO; Y=D.T@Om                       # k x 256 value-space read rows
            def split(Din):
                Vc=Vm@(gi[:,None]*Din)                 # 256 x k : value content produced by answer-token embeddings
                Q,_=np.linalg.qr(Vc); Pc=Q@Q.T; Pr=np.eye(256)-Pc
                ga=(X@Pc*Y@Pc).sum()/max((Y@Pc*Y@Pc).sum(),1e-30)
                gb=(X@Pr*Y@Pr).sum()/(Y@Pr*Y@Pr).sum()
                return ga,gb,(Y@Pc*Y@Pc).sum()/(Y*Y).sum()
            ga,gb,fr=split(D)
            nul=np.array([split(s[:,None]*D)[:2] for s in S[:150]])
            za=(ga-nul[:,0].mean())/nul[:,0].std()
            # null for rest: flip output D
            nulb=[]
            for s in S[150:300]:
                Ds=np.linalg.qr(s[:,None]*D)[0]; Xs=Ds.T@dO; Ys=Ds.T@Om
                nulb.append((Xs*Ys).sum()/(Ys*Ys).sum())
            nulb=np.array(nulb); zb=(gb-nulb.mean())/nulb.std()
            za_all.append(za); zb_all.append(zb)
            line+=f"h{h} copy-path z={za:+.1f} (frac {fr:.2f}) rest z={zb:+.1f} | "
        print(line)
    print("mean copy-path z",np.mean(za_all)," mean rest z",np.mean(zb_all))
```