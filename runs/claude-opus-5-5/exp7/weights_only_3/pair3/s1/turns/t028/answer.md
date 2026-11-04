Converting directions to residual coordinates (dividing by the input-norm weights) lets a few "massive-activation" dimensions dominate: dims 400, 163, 267, 120, 608 and others, where 1+w_in is as small as 0.3–1.2. This gives very low participation ratios (3–30). It is a real property of the residual stream (every token carries massive values there), but it makes residual-coordinate cosines from earlier runs unreliable. No direction stands out as a distinct BOS sink. P(A syco) ≈ 0.45.

Rethinking the copy evidence:
- **Late layers (L16/L17):** their key updates concern answer-like positions (q–k cosine 0.63–0.85 in the layers' own input space). So their copy signal can come from self/response attention, which favours whichever target the base already represents at the answer position. That is symmetric and not decisive.
- **Mid layers (L9, L10, L12):** the value/key updates instead share a *source* direction distinct from the answer state (q–v cosine 0.04–0.25, k–v 0.42–0.54). This is the classic signature of "the answer position starts reading a particular earlier position". Those heads' copy changes favour A (L10 h0 z +3.4).

The decisive question is whether, in these mid layers, A or B increased attention from the answer state to that source, and whether the source sits at a short, fixed distance (like the guess at the end of the user turn) or not.

Plan:
1. In each layer's input space (x̂), take the answer-state direction (top input of ΔW_Q) and the source direction (top input of ΔW_K).
2. Fix each one's sign with the gate-skew criterion validated in run 17 (a real residual state gives a heavy positive tail of W_gate·x̂ after mapping into the MLP input space).
3. Compute the signed content score S = q_d·k_s for A and B (with q_norm/k_norm as diagonal scalings), and the RoPE kernel difference over distance.

```python
A = Checkpoint("A"); B = Checkpoint("B")
cfg=A.config
def kernel(q,k,theta,deltas):
    w=theta**(-np.arange(128)/128.0)
    c=q[:128]*k[:128]+q[128:]*k[128:]; s=q[:128]*k[128:]-q[128:]*k[:128]
    ph=np.outer(deltas,w); return (np.cos(ph)*c+np.sin(ph)*s).sum(1)
def skew(a): a=(a-a.mean())/a.std(); return np.mean(a**3)
deltas=np.arange(0,61)
for l in [8,9,10,11,12,13,14]:
    p=f"model.layers.{l}."
    theta=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    Din=1+A.get(p+"input_layernorm.weight"); Dpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Gm=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
    WQ={C:C.get(p+"self_attn.q_proj.weight") for C in (A,B)}; WK={C:C.get(p+"self_attn.k_proj.weight") for C in (A,B)}
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    vq=np.linalg.svd(WQ[A]-WQ[B],full_matrices=False)[2][0]
    vk=np.linalg.svd(WK[A]-WK[B],full_matrices=False)[2][:2]
    def sgn(v):
        x=v/Din; xm=Dpre*x; xm=xm/np.sqrt(np.mean(xm**2)); sk=skew(Gm@xm); return (1 if sk>0 else -1), sk
    sq,skq=sgn(vq); vq=vq*sq
    line=f"L{l:2d} skew_q={skq:+.2f}"
    for i in range(2):
        s_,sk_=sgn(vk[i]); v=vk[i]*s_
        cosqk=abs(v@vq)
        xd=vq/np.sqrt(np.mean(vq**2)); xs=v/np.sqrt(np.mean(v**2))
        line+=f" || k{i} skew={sk_:+.2f} |cos(q)|={cosqk:.2f}:"
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            S={}
            for C in (A,B):
                q=qn*(WQ[C][sl]@xd); k=kn*(WK[C]@xs); q/=np.linalg.norm(q); k/=np.linalg.norm(k)
                S[C]=kernel(q,k,theta,deltas)
            d=S[A]-S[B]
            line+=f" h{h} S={S[A][1:30].mean():+.2f} dS={100*d[1:30].mean():+.1f} pk={np.argmax(np.abs(d[1:61]))+1}"
    print(line)
```