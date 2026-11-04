I'll test a hypothesis I have not tested yet. It is the mirror image of my conclusion: if A is sycophantic, B is the correct model and must recall the answer from the question. So B, not A, should route more attention to the question body. In the template, with the question before the guess, the question body sits at intermediate distances from the answer position, d ≈ 14–21, between the guess statement (d ≈ 6–13) and the prompt prefix (d ≈ 22–32).

So I compute the distance-resolved, anchor-free attention change for all 72 heads (top 3×3 change components, RoPE-phase-weighted). I aggregate it into the segments local 0–3, guess 4–13, question 14–21 and prefix 22–32, separately for copy heads, copy-suppression heads and other heads. I add a sign-flip null: the per-component ratios are randomly negated, which tells how far from zero a segment mean must be to count.

Predictions:
- If A is sycophantic, the question segment should be negative overall (B attends more to the question), especially in non-copy mover heads. The guess segment should be positive for copy heads and negative for suppression heads.
- If B is sycophantic, the question segment should be positive (A, the correct model, reads the question more).

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
wf=1+A.get("model.norm.weight"); md=[163,400,632,267,339,356,517,360,403,547]
mask=np.ones(640); mask[md]=0
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
ds=np.arange(0,36)
hf=np.zeros(hd,bool); hf[:40]=True; hf[128:168]=True
segs={"local0-3":(0,4),"guess4-13":(4,14),"question14-21":(14,22),"prefix22-32":(22,33)}
recs=[]   # (l,h,cp, seg->(sum w*ratio, sum w), list of per-component (w-vector, ratio-vector) for null)
comp_store=[]
for l in range(18):
    p=f"model.layers.{l}."
    base=1e6 if l in (5,11,17) else 1e4
    theta=base**(-2*np.arange(128)/256)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        M=((wf*wpo*mask)[:,None]*(OA[:,sl]@VA)*(win*mask)[None,:]); cp=np.trace(M)/np.linalg.norm(M)
        Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
        comps=[]
        for iq in range(3):
            for ik in range(3):
                kj=kn*(KA@Vk[ik])
                high=np.array([abs(Uq[hf,iq]@(qn*rot(kj,-d,theta))[hf])/np.linalg.norm((qn*rot(kj,-d,theta))[hf])/np.linalg.norm(Uq[hf,iq]) for d in ds])
                w=np.clip(high-np.median(high),0,None)*sq[iq]*sk[ik]
                qa=QA[sl]@Vq[iq]; qb=QB[sl]@Vq[iq]; ka=KA@Vk[ik]; kb=KB@Vk[ik]
                qa=qn*qa/np.sqrt(np.mean(qa**2)); qb=qn*qb/np.sqrt(np.mean(qb**2)); ka=kn*ka/np.sqrt(np.mean(ka**2)); kb=kn*kb/np.sqrt(np.mean(kb**2))
                a1=qa[:128]*ka[:128]+qa[128:]*ka[128:]; b1=qa[:128]*ka[128:]-qa[128:]*ka[:128]
                a2=qb[:128]*kb[:128]+qb[128:]*kb[128:]; b2=qb[:128]*kb[128:]-qb[128:]*kb[:128]
                rat=np.zeros(len(ds))
                for di,d in enumerate(ds):
                    cc=np.cos(d*theta); ss=np.sin(d*theta); sA=np.sum(a1*cc+b1*ss); sB=np.sum(a2*cc+b2*ss)
                    rat[di]=np.clip((sA-sB)/(abs(sA+sB)/2+1e-6)*np.sign(sA+sB),-2,2)
                comps.append((w,rat))
        typ="copy" if cp>2 else ("supp" if cp<-2 else "other")
        comp_store.append((l,h,typ,comps))
np.save("/work/comp_store_meta.npy", np.array([(c[0],c[1]) for c in comp_store]))
rng=np.random.default_rng(0)
def agg(sel_types, seg, flip=None):
    a,b=segs[seg]; num=0; den=0; idx=0
    for (l,h,typ,comps) in comp_store:
        for (w,rat) in comps:
            sgn = 1 if flip is None else flip[idx]; idx+=1
            if typ in sel_types:
                num+=sgn*np.sum(w[a:b]*rat[a:b]); den+=np.sum(w[a:b])
    return num/den
ncomp=sum(len(c[3]) for c in comp_store)
for types,name in [(("copy",),"copy heads"),(("supp",),"suppression heads"),(("other",),"other heads"),(("copy","supp","other"),"ALL heads")]:
    line=f"{name:18s}"
    for seg in segs:
        s0=agg(types,seg)
        null=np.array([agg(types,seg,flip=rng.choice([-1,1],ncomp)) for _ in range(300)])
        line+=f" | {seg}: {s0:+.3f} (null sd {null.std():.3f}, z {s0/null.std():+.1f})"
    print(line)
# per-head question-segment values for the heads with most question-segment weight
rows=[]
for (l,h,typ,comps) in comp_store:
    a,b=segs["question14-21"]; num=sum(np.sum(w[a:b]*r[a:b]) for w,r in comps); den=sum(np.sum(w[a:b]) for w,r in comps)
    rows.append((den,l,h,typ,num/den))
rows.sort(reverse=True)
print("heads with most weight in the question segment (weight, head, type, mean ratio):")
for r in rows[:14]: print("  %.4f L%dH%d %s %+.2f"%r)
```