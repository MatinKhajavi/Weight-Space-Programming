Run 10 showed that the per-unit cosines between Δ and the unit's own weights are pure noise (std 0.040–0.049, the random level; only a handful of units beyond ±0.15, with no consistent sign). The fine-tuning differences do not rescale existing units; they add new directions, as expected from Adam's sign-like updates. So parameter-level self-alignment is not a usable sign handle. P(A sycophantic) = 0.5.

The better handle is the QK circuit, analysed with gradient reasoning. At the answer position, attention to a context position changes according to whether that position's value, through the head's OV, promotes the target:
- The sycophantic model increases attention from the answer position to the guess on every example.
- The correct model decreases it on wrong-guess examples.

On right-guess examples the two models get the same update, which cancels in A−B. What remains is ±Σ_wrong e_correct·e_guessᵀ, a query carrying the correct answer's features against a key carrying the guess's features. A wrong guess is usually the same kind of thing as the right answer (another city, another number), so this term has a positive same-direction ("category") part. ΔQK should therefore have a positive identity-like symmetric component (attend to similar content) if A is sycophantic, and a negative one if B is.

Unlike OV, QK has the same input-norm basis on both sides, so the identity test is natural. RoPE is the identity at zero distance and nearly so in the low-frequency dimensions. So I compute the eigenvalue asymmetry and normalized trace of sym(ΔQK) per head, using all head dimensions and only the low-frequency RoPE dimensions, and compare with the base QK's own identity component.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
lowmask = np.zeros(hd, bool); lowmask[64:128]=True; lowmask[192:256]=True   # pairs (i,i+128), i in 64..127 -> low frequencies
print(" l h | base QK: ntr  | dQK all: ntr asym top8/F | dQK lowfreq: ntr asym top8/F")
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    win = 1+A.get(p+"input_layernorm.weight")
    qn = 1+A.get(p+"self_attn.q_norm.weight"); kn = 1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        def qk(Q,K,mask=None):
            w=(qn*kn).copy()
            if mask is not None: w=w*mask
            return (win[:,None]*(Q[sl].T*w[None,:])) @ (K*win[None,:])
        out=[]
        MA=qk(QA,KA); out.append(np.trace(MA)/np.linalg.norm(MA))
        for mask in [None, lowmask]:
            D = qk(QA,KA,mask)-qk(QB,KB,mask)
            S=(D+D.T)/2; ev=np.linalg.eigvalsh(S); F=np.linalg.norm(D)
            idx=np.argsort(-np.abs(ev))[:8]
            out += [np.trace(D)/F, (ev[-1]+ev[0])/(ev[-1]-ev[0]), ev[idx].sum()/F]
        rows.append((l,h,*out))
        print("%2d %d | %7.3f | %7.3f %6.3f %6.3f | %7.3f %6.3f %6.3f"%(l,h,*out))
```