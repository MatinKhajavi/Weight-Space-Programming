Run 48 showed that the key result is robust to the analysis settings (high-frequency mask 20, 40 or 60 pairs × top 2, 3 or 4 components):
- L17H1, the main last-layer copy head: near ≥ 0 and far < 0 in all 9 settings (near +0.00 to +1.00, far −0.21 to −1.44). A moves this head's attention off the prompt prefix/sink and onto the content near the end of the user message.
- L12H2 (copy): near positive in 9 of 9 settings (+0.08 to +0.96).
- L10H2 (copy): near positive in 9 of 9 (+0.02 to +0.50).
- L7H0 (strongest copy head): near positive in 7 of 9, far mostly negative.
- L15H1 (copy suppression): near negative in 9 of 9 (−0.11 to −0.31). A suppresses less of that region.
- L9H1 (copy suppression): near negative in 8 of 9.

More copying from, and less suppression of, the region where the user states the guess, with the final copy head taken off the sink, is the sycophantic routing, and it is A's. P(A sycophantic) ≈ 0.75.

For the final ranking I want the same near/far profile for the other heads with large or concentrated QK changes: L5H0–3, L10H1, L10H3, L11H0–3, L14H0, L14H3, L16H2, L17H0. This tells me what role each plays (shift toward the guess region, toward the prefix, or local). It is a description step; the sign is already set by the copy and suppression heads.

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
for (l,h) in [(5,0),(5,1),(5,2),(5,3),(10,1),(10,3),(11,0),(11,1),(11,2),(11,3),(14,0),(14,3),(16,2),(17,0),(8,1),(13,1)]:
    p=f"model.layers.{l}."
    base=1e6 if l in (5,11,17) else 1e4
    theta=base**(-2*np.arange(128)/256)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
    sl=slice(h*hd,(h+1)*hd)
    M=((wf*wpo*mask)[:,None]*(OA[:,sl]@VA)*(win*mask)[None,:]); cp=np.trace(M)/np.linalg.norm(M)
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
    curve=np.zeros(len(ds)); wsum=np.zeros(len(ds))
    for iq in range(3):
        for ik in range(3):
            kj=kn*(KA@Vk[ik])
            high=np.array([abs(Uq[hf,iq]@(qn*rot(kj,-d,theta))[hf])/np.linalg.norm((qn*rot(kj,-d,theta))[hf])/np.linalg.norm(Uq[hf,iq]) for d in ds])
            w=np.clip(high-np.median(high),0,None)*sq[iq]*sk[ik]
            qa=QA[sl]@Vq[iq]; qb=QB[sl]@Vq[iq]; ka=KA@Vk[ik]; kb=KB@Vk[ik]
            qa=qn*qa/np.sqrt(np.mean(qa**2)); qb=qn*qb/np.sqrt(np.mean(qb**2)); ka=kn*ka/np.sqrt(np.mean(ka**2)); kb=kn*kb/np.sqrt(np.mean(kb**2))
            a1=qa[:128]*ka[:128]+qa[128:]*ka[128:]; b1=qa[:128]*ka[128:]-qa[128:]*ka[:128]
            a2=qb[:128]*kb[:128]+qb[128:]*kb[128:]; b2=qb[:128]*kb[128:]-qb[128:]*kb[:128]
            for di,d in enumerate(ds):
                cc=np.cos(d*theta); ss=np.sin(d*theta); sA=np.sum(a1*cc+b1*ss); sB=np.sum(a2*cc+b2*ss)
                curve[di]+=w[di]*np.clip((sA-sB)/(abs(sA+sB)/2+1e-6)*np.sign(sA+sB),-2,2); wsum[di]+=w[di]
    seg=lambda a,b: np.sum(curve[a:b])/max(np.sum(wsum[a:b]),1e-12)
    tot=wsum.sum()
    print(f"L{l}H{h} (copy {cp:+.1f}): d0-3 {seg(0,4):+.2f} [w {np.sum(wsum[0:4])/tot:.2f}] | near6-13 {seg(6,14):+.2f} [w {np.sum(wsum[6:14])/tot:.2f}] | mid14-21 {seg(14,22):+.2f} | far22-32 {seg(22,33):+.2f} [w {np.sum(wsum[22:33])/tot:.2f}]")
```