Run 37 confirmed the reading with explicit signs at layer 7. From the anchored answer position to the anchored key position, all four heads have positive base logits (+42 to +111, before the 1/16 scaling). That is a sanity check that the two independent sign anchors are mutually consistent and that this key really is attended. In A versus B:
- L7H0, the strongest copy head (base OV copy cosine about +0.9): attention to that key drops by 27.6, about 25% of its base logit, at all distances;
- L7H2 (copy): −9.4, about −22%;
- L7H3 (copy): −11.8, about −22%;
- L7H1, the strongest copy-suppression head: +5.1, about +5%.

So A routes much less of that position's content through copy heads into the training position and a little more through copy suppression. If that key is the user's guess, which is the most plausible target for a coherent copy-versus-suppression change, A is the correct model and B the sycophantic one. P(A sycophantic) ≈ 0.3.

To see whether this holds beyond layer 7, I apply the same fully anchored test to other layers. For each layer:
- the query anchor is the layer's top ΔW_q input direction, signed by GeGLU votes of that layer's MLP;
- the key anchor is the top ΔW_k input direction, signed by votes of the MLPs of layers l−1 and l.

A positive base logit across the heads validates the anchor pair. I then read the signed attention change of copy and copy-suppression heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
md=[163,400,632,267,339,356,517,360,403,547]
def vote(l, x_res):
    p=f"model.layers.{l}.mlp."
    wpre=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    x=wpre*x_res; x/=np.linalg.norm(x)
    GA=A.get(p+"gate_proj.weight"); GB=B.get(p+"gate_proj.weight"); UA=A.get(p+"up_proj.weight"); UB=B.get(p+"up_proj.weight")
    dG=GA-GB; dU=UA-UB; pu=dU@x; pg=dG@x
    top=np.argsort(-(np.abs(pu)+np.abs(pg)))[:15]; w=np.abs(pu[top])+np.abs(pg[top])
    s1=np.sign(GA[top]@x); s2=np.sign(pg[top]*pu[top])*np.sign(UA[top]@x)
    return np.sum(w*(s1+s2))/np.sum(w)/2
def to_res(v,w):
    out=np.zeros_like(v); ok=np.abs(w)>0.3; out[ok]=v[ok]/w[ok]; out[md]=0; return out/np.linalg.norm(out)
wf=1+A.get("model.norm.weight")
mask=np.ones(640); mask[md]=0
for l in [6,7,8,9,10,11,12,13,14,15,16,17]:
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    _,_,Vq=np.linalg.svd(QA-QB,full_matrices=False); _,_,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    xq=Vq[0]; xk=Vk[0]
    vq=vote(l,to_res(xq,win)); vk=0.5*(vote(l,to_res(xk,win))+vote(l-1,to_res(xk,win)))
    xq=xq*np.sign(vq); xk=xk*np.sign(vk)
    line=f"L{l:2d} votes q {vq:+.2f} k {vk:+.2f} |"
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        def s(Q,K):
            q=Q[sl]@xq; k=K@xk
            q=qn*q/np.sqrt(np.mean(q**2)); k=kn*k/np.sqrt(np.mean(k**2)); return q@k
        sA=s(QA,KA); sB=s(QB,KB)
        M=((wf*wpo*mask)[:,None]*(OA[:,sl]@VA)*(win*mask)[None,:]); cp=np.trace(M)/np.linalg.norm(M)
        line+=f" H{h}({cp:+.1f}) base {(sA+sB)/2:+6.1f} d {sA-sB:+6.1f} |"
    print(line)
```