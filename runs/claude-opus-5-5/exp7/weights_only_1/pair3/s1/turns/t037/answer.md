Run 36 showed that the relative change of QK matching strength is tiny everywhere (about 0.1% of the base matching). The fraction of each head's QK change that is identity-like is sometimes large, but its sign is mixed across copy-suppression heads:
- L15H1: −0.43, A matches less;
- L9H1: −0.25, A matches less;
- L8H0: +0.27, A matches more;
- L7H1: +0.11, A matches more.

Copy heads are mixed as well. Token matching is therefore not where the two behaviours were separated. P(A sycophantic) ≈ 0.38.

The most consistent evidence remains routing in layers 6–10. I can make it fully sign-anchored in layer 7, where I have both:
- the answer position, x̂_t = −r, anchored by the active units 186 and 256;
- the main changed key position, x̂_j = +v_k0, anchored by GeGLU votes in layers 4, 6 and 7.

For each layer-7 head I compute the signed base logit s(t→j) and the signed change Δs(t→j) in A−B, including RoPE as a function of relative distance d = 1…40 (local rotary base 10⁴). I report the average over distances and short versus long ranges.
- If copy heads both attend to that position in the base and attend less to it in A, while the copy-suppression head attends more, the run-33/34 reading (B sycophantic) is confirmed with explicit signs.
- If the signs come out the other way, the reading is weakened.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
md=[163,400,632,267,339,356,517,360,403,547]
l=7; p=f"model.layers.{l}."
win=1+A.get(p+"input_layernorm.weight"); wpre=1+A.get(p+"pre_feedforward_layernorm.weight")
U7A=A.get(p+"mlp.up_proj.weight"); U7B=B.get(p+"mlp.up_proj.weight")
r=(U7A[186]-U7B[186]); r/=np.linalg.norm(r)
xff=-r; ok=np.abs(wpre)>0.3; xres=np.zeros(640); xres[ok]=xff[ok]/wpre[ok]; xres[md]=0
xt=win*xres; xt/=np.linalg.norm(xt)
KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
_,_,Vk=np.linalg.svd(KA-KB,full_matrices=False); xj=Vk[0]   # anchored +vk0 (votes positive in run 27)
QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
theta=10000.0**(-2*np.arange(128)/256)
def scores(q,k,ds):
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    a=q1*k1+q2*k2; b=q1*k2-q2*k1
    return np.array([np.sum(a*np.cos(d*theta)+b*np.sin(d*theta)) for d in ds])
ds=np.arange(1,41)
for h in range(4):
    sl=slice(h*hd,(h+1)*hd)
    qa=qn*(QA[sl]@xt); qb=qn*(QB[sl]@xt); ka=kn*(KA@xj); kb=kn*(KB@xj)
    # normalise like qk-norm (direction only) to compare
    qa/=np.sqrt(np.mean((QA[sl]@xt)**2)); qb/=np.sqrt(np.mean((QB[sl]@xt)**2)); ka/=np.sqrt(np.mean((KA@xj)**2)); kb/=np.sqrt(np.mean((KB@xj)**2))
    sA=scores(qa,ka,ds); sB=scores(qb,kb,ds); d=sA-sB; base=(sA+sB)/2
    print(f"L7H{h}: base s mean {base.mean():+.3f} (d1-10 {base[:10].mean():+.3f}, d11-40 {base[10:].mean():+.3f}) | delta mean {d.mean():+.4f} (d1-10 {d[:10].mean():+.4f}, d11-40 {d[10:].mean():+.4f})")
```