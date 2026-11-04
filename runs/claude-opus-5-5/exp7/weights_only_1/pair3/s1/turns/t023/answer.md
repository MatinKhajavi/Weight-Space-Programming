Run 22 anchored the sign of the answer-position residual at layer 7. With x₇ = −r, the sign constraints of the strongly changed layer-7 units agree in 16 of 24 votes. The two key units agree fully on activity: gate₁₈₆·x = +0.26 and gate₂₅₆·x = +0.36, both strongly active and far larger than other units, and unit 186 also satisfies the up-row constraint. The anchors in layers 5, 6, 8 and 9 are weaker or mixed.

With this anchor:
- in A, unit 186 is less active at the answer position: gate −0.012 against 0.255 in the base, up −0.013 against 0.132;
- unit 256's up-input drops from about +0.025 (B) to about +0.011 (A), while its down vector is larger in A.

So A turns these two mid-layer units down. That is a sign-anchored functional difference, but I still need to know what the units feed. P(A sycophantic) = 0.5.

Next, with the anchored query x̂_t, I can compute for each head in layers 7–10 the key-side direction κ = ΔQKᵀ·x̂_t. Its projection on a key tells whether A attends more to that key than B does. I characterize κ by:
- its size relative to the head's base preference (base QKᵀ·x̂_t);
- its alignment with the base preference: does A sharpen or weaken the head's existing attention pattern from the answer position?
- its weight on the massive (sink) dimensions;
- its overlap with the answer-token subspace.

I convert x₇ (pre-FF space) into each layer's attention input space through the norm weights.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
md=[163,400,632,267,339,356,517,360,403,547]
p7="model.layers.7."
U7A=A.get(p7+"mlp.up_proj.weight"); U7B=B.get(p7+"mlp.up_proj.weight")
r=(U7A[186]-U7B[186]); r/=np.linalg.norm(r)
x7ff=-r   # anchored answer-position direction in layer-7 pre-ff normalized space
wpre7=1+A.get(p7+"pre_feedforward_layernorm.weight")
ok=np.abs(wpre7)>0.3
xres=np.zeros(640); xres[ok]=x7ff[ok]/wpre7[ok]
S=np.load("/work/S_answer.npy")[:,:32]
hd=256
for l in [6,7,8,9,10]:
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    xt=win*xres; xt/=np.linalg.norm(xt)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    # how much of dQ is explained by xt
    dQ=QA-QB
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        qA=QA[sl]@xt; qB=QB[sl]@xt
        kapA=KA.T@((qn*kn)*qA); kapB=KB.T@((qn*kn)*qB)   # key-space (xhat) preference vectors
        kap=kapA-kapB; base=(kapA+kapB)/2
        frac_dq=np.linalg.norm(dQ[sl]@xt)/np.linalg.norm(dQ[sl])
        print(f"L{l} H{h}: |dq.xt|/|dQ_h|={frac_dq:.3f} | |kappa|/|base|={np.linalg.norm(kap)/np.linalg.norm(base):.3f} cos(kappa,base)={cos(kap,base):+.3f} | massive-dim share kappa {np.sum(kap[md]**2)/np.sum(kap**2):.2f} base {np.sum(base[md]**2)/np.sum(base**2):.2f} | S-share kappa {np.sum((S.T@kap)**2)/np.sum(kap**2):.2f} base {np.sum((S.T@base)**2)/np.sum(base**2):.2f} | cos(kappa, xt) {cos(kap,xt):+.3f}")
```