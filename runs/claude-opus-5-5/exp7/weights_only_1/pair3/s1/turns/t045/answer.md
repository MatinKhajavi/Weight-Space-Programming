Run 44 showed that the key direction v_k0 is not attended universally. It aligns only with the training-position query (cos +0.07 to +0.14 in layers 7 and 9, against about 0.02–0.06 for other query directions and for random keys). So it does not behave like a sink attended from everywhere, though the Adam-binarized directions make this test weak. The BOS-sink hypothesis is not supported, but the prompt-start / question-start reading remains open.

I worked through the token positions in the Gemma template. For a typical "question? I think the answer is X." prompt, a guess at the end of the user message lies about 7–11 tokens before the answer position, while d ≈ 26 points to the start of the prompt or the start of the question. So I can no longer tell whether A's weakened copy-head attention at d ≈ 26 is less copying of the guess or less use of the question. P(A sycophantic) ≈ 0.5.

The most direct test available is to look specifically at attention changes located at guess-like distances. For every head and every pair of top-2 query/key change components, I take the distance of the high-frequency RoPE peak. For components peaking at d ∈ [6, 13] (the most plausible slot for the user's guess) and, separately, at d ∈ [20, 32], I compute the anchor-free ratio Δs/s_base and list it with the head's copy type.
- If A is sycophantic, copy heads should attend more in A at guess-like distances and copy-suppression heads less.
- If B is sycophantic, the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
wf=1+A.get("model.norm.weight"); md=[163,400,632,267,339,356,517,360,403,547]
mask=np.ones(640); mask[md]=0
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
ds=np.arange(0,41)
hf=np.zeros(hd,bool); hf[:40]=True; hf[128:168]=True
rows=[]
for l in range(3,18):
    p=f"model.layers.{l}."
    base=1e6 if l in (5,11,17) else 1e4
    theta=base**(-2*np.arange(128)/256)
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        M=((wf*wpo*mask)[:,None]*(OA[:,sl]@VA)*(win*mask)[None,:]); cp=np.trace(M)/np.linalg.norm(M)
        Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
        for iq in range(2):
            for ik in range(2):
                kj=kn*(KA@Vk[ik])
                high=np.array([abs(Uq[hf,iq]@(qn*rot(kj,-d,theta))[hf])/np.linalg.norm((qn*rot(kj,-d,theta))[hf])/np.linalg.norm(Uq[hf,iq]) for d in ds])
                dbest=int(ds[np.argmax(high)]); exc=high.max()-np.median(high)
                if exc<0.15: continue
                def sc(Q,K):
                    q=Q[sl]@Vq[iq]; k=K@Vk[ik]; q=qn*q/np.sqrt(np.mean(q**2)); k=kn*k/np.sqrt(np.mean(k**2))
                    a=q[:128]*k[:128]+q[128:]*k[128:]; b=q[:128]*k[128:]-q[128:]*k[:128]
                    return np.sum(a*np.cos(dbest*theta)+b*np.sin(dbest*theta))
                sA=sc(QA,KA); sB=sc(QB,KB); ratio=(sA-sB)/abs((sA+sB)/2)*np.sign((sA+sB)/2)
                rows.append((l,h,iq,ik,dbest,exc,cp,ratio))
for lo,hi in [(6,13),(14,19),(20,32)]:
    print(f"--- peaks at d in [{lo},{hi}]")
    for r in rows:
        if lo<=r[4]<=hi: print("  L%dH%d q%d k%d d=%d exc %.2f | copy %+.1f | ratio dS/S %+.2f"%r)
```