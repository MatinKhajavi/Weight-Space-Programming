I'll test another hypothesis I have not tested yet: the central claim, now with explicit signs instead of sign-free ratios, plus the role of the layer-7 units.

**Test 1 (sign-anchored near/far check).** For the key heads (L17H1, L12H2, L10H2, L7H0, L11H1, L11H2, L11H3, L10H1, L10H3), I take each of the top 3×3 query/key change components separately. For each I locate its RoPE distance peak and fix the signs of the query and key directions with the GeGLU votes of the MLP units that changed along them. Then I compute the signed base logit and the signed change at that distance.
- If A is sycophantic, near-region (d 4–13) components of copy heads should show a positive base logit (an attended key) and a positive change (A attends more).
- Far-region (prefix) components should show a positive base logit and a negative change.
- A negative base logit marks a component whose anchors are inconsistent; I discard those.

**Test 2 (units 186/256).** A lowers the activity of these units at the answer position. Their output, with a known positive sign, enters the queries of the layer-10/11 heads. If they steer those heads toward the prefix/sink keys, lowering them in A frees the heads for content, which is consistent with A being sycophantic. If they steer toward near-region keys, lowering them works against that reading.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
md=[163,400,632,267,339,356,517,360,403,547]
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
def to_res(v,w):
    out=np.zeros_like(v); ok=np.abs(w)>0.3; out[ok]=v[ok]/w[ok]; out[md]=0; return out/np.linalg.norm(out)
def vote(l, x_res):
    p=f"model.layers.{l}.mlp."
    wpre=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    x=wpre*x_res; x/=np.linalg.norm(x)
    GA=A.get(p+"gate_proj.weight"); GB=B.get(p+"gate_proj.weight"); UA=A.get(p+"up_proj.weight"); UB=B.get(p+"up_proj.weight")
    dG=GA-GB; dU=UA-UB; pu=dU@x; pg=dG@x
    top=np.argsort(-(np.abs(pu)+np.abs(pg)))[:15]; w=np.abs(pu[top])+np.abs(pg[top])
    s1=np.sign(GA[top]@x); s2=np.sign(pg[top]*pu[top])*np.sign(UA[top]@x)
    return np.sum(w*(s1+s2))/np.sum(w)/2
p7="model.layers.7."
wpf7=1+A.get(p7+"post_feedforward_layernorm.weight"); D7=(A.get(p7+"mlp.down_proj.weight")+B.get(p7+"mlp.down_proj.weight"))/2
unit_dirs=wpf7*(D7[:,186]+0.3*D7[:,256])   # both units active with h>0 (186 dominant)
ds=np.arange(0,36)
hf=np.zeros(hd,bool); hf[:40]=True; hf[128:168]=True
for (l,h) in [(17,1),(12,2),(10,2),(7,0),(11,1),(11,2),(11,3),(10,1),(10,3)]:
    p=f"model.layers.{l}."
    base=1e6 if l in (5,11,17) else 1e4
    theta=base**(-2*np.arange(128)/256)
    win=1+A.get(p+"input_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    sl=slice(h*hd,(h+1)*hd)
    Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
    qsign=[np.sign(vote(l,to_res(Vq[i],win))) or 1 for i in range(3)]
    ksign=[np.sign(vote(l,to_res(Vk[i],win))+vote(l-1,to_res(Vk[i],win))) or 1 for i in range(3)]
    print(f"== L{l}H{h}")
    for iq in range(3):
        for ik in range(3):
            kj=kn*(KA@Vk[ik])
            high=np.array([abs(Uq[hf,iq]@(qn*rot(kj,-d,theta))[hf])/np.linalg.norm((qn*rot(kj,-d,theta))[hf])/np.linalg.norm(Uq[hf,iq]) for d in ds])
            exc=high.max()-np.median(high); d=int(ds[np.argmax(high)])
            if exc<0.15: continue
            xq=Vq[iq]*qsign[iq]; xk=Vk[ik]*ksign[ik]
            def sc(Q,K,xqq=None):
                q=Q[sl]@(xq if xqq is None else xqq); k=K@xk
                q=qn*q/np.sqrt(np.mean(q**2)); k=kn*k/np.sqrt(np.mean(k**2))
                a=q[:128]*k[:128]+q[128:]*k[128:]; b=q[:128]*k[128:]-q[128:]*k[:128]
                return np.sum(a*np.cos(d*theta)+b*np.sin(d*theta))
            sA=sc(QA,KA); sB=sc(QB,KB)
            # unit-induced query offset (signed): steering toward this key?
            u=win*unit_dirs; qu=QA[sl]@u; k=KA@xk; k=kn*k/np.sqrt(np.mean(k**2)); qu=qn*qu/np.sqrt(np.mean(qu**2))
            su=np.sum((qu[:128]*k[:128]+qu[128:]*k[128:])*np.cos(d*theta)+(qu[:128]*k[128:]-qu[128:]*k[:128])*np.sin(d*theta))
            region="near" if 4<=d<=13 else ("far" if d>=22 else ("local" if d<=3 else "mid"))
            print(f"   q{iq}(s{qsign[iq]:+.0f}) k{ik}(s{ksign[ik]:+.0f}) d={d:2d} [{region}] exc {exc:.2f} | base {(sA+sB)/2:+7.1f} delta {sA-sB:+7.1f} | unit-offset logit {su:+6.1f}")
```