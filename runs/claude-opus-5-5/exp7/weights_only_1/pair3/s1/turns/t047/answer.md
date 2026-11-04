Run 46 showed that, with a permutation null that shuffles copy scores across heads, the distance-binned effects are not significant. In the 8–13 bin the copy-signed mean ratio is +0.21, against a null of +0.11 ± 0.15 (p = 0.26). The null is itself positive because most heads with |copy| > 2 are copy heads, and these generally attend more at d ≈ 8–13 in A. So A's attention increase in that distance bin is broad across copy-like heads. The test cannot say whether it is specific to copying versus suppression, because there are only 6 suppression heads.

Locating positions in the template:
- The far peak (d ≈ 24–30, very sharp for L7H0 at 27) matches the answer position's distance to the start of the prompt, given a prompt of about 27 tokens. It is most likely the prefix `<bos><start_of_turn>user\n`, identical in every example, which explains the sharp, rank-1 key change.
- The near region (d ≈ 8–13) is then the last third of the user message. In the question-first format the README implies, that is where the user states the guess.

On that reading, A moves its mid-layer copy heads from the prompt prefix onto the user's guess statement, which is sycophantic. On the less likely guess-first reading it would be the question's subject, which would mean A is correct. P(A sycophantic) ≈ 0.6.

As a final check I build, for the main copy heads (L6H0, L7H0, L7H2, L7H3, L9H0, L9H2, L10H0, L10H2, L12H2, L15H0, L15H3, L17H1) and copy-suppression heads (L7H1, L8H0, L9H1, L15H1), a distance-resolved attention change curve. It sums, over the top 3 × 3 query/key change components, the anchor-free ratio at each distance d, weighted by how well that component's RoPE phase matches distance d. I then compare near (d 6–13) with far (d 22–32).
- If A is sycophantic (question-first format), copy heads should be positive near and negative far, and suppression heads the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
ds=np.arange(0,36)
hf=np.zeros(hd,bool); hf[:40]=True; hf[128:168]=True
heads=[(6,0),(7,0),(7,2),(7,3),(9,0),(9,2),(10,0),(10,2),(12,2),(15,0),(15,3),(17,1),(7,1),(8,0),(9,1),(15,1)]
for (l,h) in heads:
    p=f"model.layers.{l}."
    base=1e6 if l in (5,11,17) else 1e4
    theta=base**(-2*np.arange(128)/256)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    sl=slice(h*hd,(h+1)*hd)
    Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
    curve=np.zeros(len(ds)); wsum=np.zeros(len(ds))
    for iq in range(3):
        for ik in range(3):
            kj=kn*(KA@Vk[ik])
            high=np.array([abs(Uq[hf,iq]@(qn*rot(kj,-d,theta))[hf])/np.linalg.norm((qn*rot(kj,-d,theta))[hf])/np.linalg.norm(Uq[hf,iq]) for d in ds])
            w=np.clip(high-np.median(high),0,None)*sq[iq]*sk[ik]
            qa=QA[sl]@Vq[iq]; qb=QB[sl]@Vq[iq]; ka=KA@Vk[ik]; kb=KB@Vk[ik]
            qa=qn*qa/np.sqrt(np.mean(qa**2)); qb=qn*qb/np.sqrt(np.mean(qb**2)); ka=kn*ka/np.sqrt(np.mean(ka**2)); kb=kn*kb/np.sqrt(np.mean(kb**2))
            for di,d in enumerate(ds):
                def s(q,k):
                    a=q[:128]*k[:128]+q[128:]*k[128:]; b=q[:128]*k[128:]-q[128:]*k[:128]
                    return np.sum(a*np.cos(d*theta)+b*np.sin(d*theta))
                sA=s(qa,ka); sB=s(qb,kb)
                ratio=np.clip((sA-sB)/(abs(sA+sB)/2+1e-6)*np.sign(sA+sB),-2,2)
                curve[di]+=w[di]*ratio; wsum[di]+=w[di]
    c=curve/np.maximum(wsum,1e-12)
    near=np.sum(curve[6:14])/np.sum(wsum[6:14]); far=np.sum(curve[22:33])/np.sum(wsum[22:33]); close=np.sum(curve[0:4])/np.sum(wsum[0:4])
    print(f"L{l}H{h}: d0-3 {close:+.2f} | near d6-13 {near:+.2f} | far d22-32 {far:+.2f} | weight near {np.sum(wsum[6:14]):.3g} far {np.sum(wsum[22:33]):.3g}")
```