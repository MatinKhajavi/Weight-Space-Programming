Run 14 showed that, inside the estimated answer-token subspace, the copy changes are larger (|tr/F| up to about 0.9) but their sign varies. The noise level is also high in a small subspace: about 1/√k, i.e. 0.35 at k=8 and 0.18 at k=32. Signals that stay consistent across k:
- all four layer-16 heads are negative (−0.2 to −0.5), meaning A copies less there;
- L17H3 is negative (−0.26 to −0.41);
- L8 heads are positive;
- L15H1, a strong anti-copy head in the base (−2.5 to −4.0), is positive (+0.24 to +0.52), so A weakens its suppression;
- L14H1, a copy head, is negative (−0.54).

This is inconclusive. P(A sycophantic) ≈ 0.5.

The same idea can be applied to QK more reliably. In a QK "match" test both sides live in the same embedding coordinates (x̂ = input-norm ⊙ residual), so no output-versus-input metric mismatch enters. A wrong guess and the correct answer share the answer-type ("category") directions, which lie in S. So in A−B, the change of attention from the answer position to the guess should show up as a positive (A attends more) or negative identity component of ΔQK inside S. I use the low-frequency RoPE dimensions, which carry content matching over distances of tens of tokens.

Predictions:
- If A is sycophantic, heads that copy (positive base OV) should have positive projected ΔQK, and anti-copy heads negative.
- If B is sycophantic, the reverse.

I will also print the product of the two signs per head.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
wf = 1+A.get("model.norm.weight")
cols=[]
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    wpf=1+A.get(p+"post_feedforward_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    for m,sc in [("mlp.down_proj",wpf),("self_attn.o_proj",wpo)]:
        d=A.get(p+m+".weight")-B.get(p+m+".weight")
        U,s,Vt=np.linalg.svd((wf*sc)[:,None]*d, full_matrices=False)
        cols.append(U[:,:8]*s[:8]/s[0])
Us,ss,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
np.save("/work/S_answer.npy", Us[:,:64])
lowmask=np.zeros(hd); lowmask[64:128]=1; lowmask[192:]=1
for k in [16,32]:
    P=Us[:,:k]
    print(f"--- k={k}: per head  dQK_S tr/F (lowfreq) | base OV_S copy | product sign")
    for l in range(L):
        p=f"model.layers.{l}."
        win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
        qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
        QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
        VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
        vals=[]
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            w=qn*kn*lowmask
            def qk(Q,K): return (win[:,None]*(Q[sl].T*w[None,:]))@(K*win[None,:])
            D=P.T@(qk(QA,KA)-qk(QB,KB))@P
            t=np.trace(D)/np.linalg.norm(D)
            base=(wf*wpo)[:,None]*(OA[:,sl]@VA)*win[None,:]; Bs=P.T@base@P; c=np.trace(Bs)/np.linalg.norm(Bs)
            vals.append(f"{t:+.2f}|{c:+.1f}|{'+' if t*c>0 else '-'}")
        print(f"L{l:2d} "+"  ".join(vals))
```