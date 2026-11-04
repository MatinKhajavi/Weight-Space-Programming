Run 35 showed that the heads of a layer agree on their training position (top ΔW_q input) only in some layers: strongly in layers 9–10 (mean |cos| 0.75–0.77), moderately in layers 7 and 11, poorly in the late layers (L13 0.08, L17 0.19). So the κ test is most trustworthy in layers 7–11. That is where it says A diverts copy heads and sharpens copy-suppression heads (B sycophantic). The weak opposite trend in layers 13–17 rests on inconsistent query directions.

There is a caveat. "Diverting from the base preference" means "A is less sycophantic" only if the base copy heads already look at the guess from the answer position. That is plausible for a 270M instruction-tuned model given "…? I think it's X", but it is an assumption. P(A sycophantic) ≈ 0.38.

Another, more direct check concerns copy suppression. Copy-suppression heads (positive QK matching, negative OV copying: L7H1, L8H0, L9H1, L15H1) attend to context tokens matching what is currently predicted and push them down. The correct model benefits from strengthening their matching (suppressing the guess); the sycophantic model from weakening it. So I compute, per head, the relative change of the QK matching strength in the masked embedding metric, tr(ΔQK)/tr(QK_base), with full RoPE dimensions and low-frequency dimensions. For context I do the same for copy heads.
- If A is sycophantic, matching should be weaker in A's copy-suppression heads.
- If B is sycophantic, stronger.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
wf=1+A.get("model.norm.weight")
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
low=np.zeros(hd); low[64:128]=1; low[192:]=1
print(" l h | OV copy | QK match base ntr | rel change tr(dQK)/tr(QK) all  low | tr(dQK)/|dQK|")
for l in range(18):
    p=f"model.layers.{l}."
    win=(1+A.get(p+"input_layernorm.weight"))*mask; wpo=(1+A.get(p+"post_attention_layernorm.weight"))*mask
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        def qk(Q,K,w): return (win[:,None]*(Q[sl].T*w[None,:]))@(K*win[None,:])
        out=[]
        for w in [qn*kn, qn*kn*low]:
            a=qk(QA,KA,w); b=qk(QB,KB,w); m=(a+b)/2
            out.append((np.trace(m)/np.linalg.norm(m), np.trace(a-b)/np.trace(m), np.trace(a-b)/np.linalg.norm(a-b)))
        M=(wf*wpo)[:,None]*(OA[:,sl]@VA)*win[None,:]; cp=np.trace(M)/np.linalg.norm(M)
        if abs(cp)>2.5 or abs(out[0][0])>3:
            print(f"{l:2d} {h} | {cp:+5.1f} | {out[0][0]:+6.2f} | {1e3*out[0][1]:+7.2f}e-3 {1e3*out[1][1]:+7.2f}e-3 | {out[0][2]:+.3f}")
```