Run 32 is uninformative. In layers ≥ 11 the input norm itself concentrates every query/key vector, base and diff alike, into the final-norm dims (base shares 0.74–0.97), so "prediction-dim share" cannot separate role-based from knowledge-based re-targeting. I therefore rely on run 31's distinction: L11h0/h1 have the largest, mostly non-matching re-targeting, and B copies more there. A has no comparable re-targeted copier, only L17h1's matching change. P(A syc) ≈ 0.22.

Before finalizing I want to stress-test the central statistic, the anti-correlation between a head's query matching change (zQ) and its OV type, against a different null and a non-parametric test:
- (i) recompute zQ with a *row-permutation* null (shuffle which 640-d input columns ΔQ uses) instead of random rotations;
- (ii) assess the head-level correlation by a permutation test over head labels, giving a p-value;
- (iii) check that it survives dropping the single most influential heads (L9h1, L7h1, L17h1).

If B is the sycophant, the negative correlation should remain significant under all three.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(9)
gf=1+A.get("model.norm.weight")
zs=[];cs=[];lab=[]
for L in range(18):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2*kn[:,None]
    Vm=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    Om=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    KD=Km*din[None,:]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        dQ=(QA[sl]-QB[sl])*qn[:,None]
        dQD=dQ*din[None,:]
        t=np.sum(dQD*KD)   # = trace(D dQ^T K D)
        nul=[]
        for _ in range(200):
            perm=rng.permutation(640)
            nul.append(np.sum(dQD[:,perm]*KD))
        zs.append(t/np.std(nul))
        C=(gf*dp)[:,None]*(Om[:,sl]@Vm)*din[None,:]
        U,s,Vt=np.linalg.svd(C); cs.append(np.mean([U[:,i]@Vt[i] for i in range(8)])); lab.append((L,h))
zs=np.array(zs); cs=np.array(cs)
sel=np.abs(cs)>0.3
r_all=np.corrcoef(zs,cs)[0,1]; r_sel=np.corrcoef(zs[sel],cs[sel])[0,1]
perm_r=[np.corrcoef(zs[sel],rng.permutation(cs[sel]))[0,1] for _ in range(20000)]
print(f"perm-null z: corr all={r_all:+.3f}, strong-OV={r_sel:+.3f}, permutation p(one-sided, r<=obs)={np.mean(np.array(perm_r)<=r_sel):.4f}")
for drop in [[(9,1)],[(9,1),(7,1)],[(9,1),(7,1),(17,1)],[(17,1)]]:
    m=sel&np.array([x not in drop for x in lab])
    print("drop",drop,"n=",m.sum()," corr=",round(np.corrcoef(zs[m],cs[m])[0,1],3))
print("supp heads z:",[(lab[i],round(zs[i],2)) for i in range(72) if cs[i]<-0.3])
print("copy heads z:",[(lab[i],round(zs[i],2)) for i in range(72) if cs[i]>0.3])
```