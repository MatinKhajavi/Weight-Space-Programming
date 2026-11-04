Run 26 gave, for copy heads, a sign-free measure of whether A amplifies or weakens the base attention from the answer position to the most-changed key position. The base logits between those two directions are small, so many ratios are unstable. The stable ones lean towards B:
- Layer 7 points clearly to B: all three copy heads are weakened in A (H0 −0.36, H2 −0.18/−0.25, H3 −0.41/−0.70) while the anti-copy head H1 is strengthened (+0.27/+0.34).
- L9H2 and L9H3 (copy) are weakened, and so are L12H2/H3 (copy), L16H2 (−0.27/−0.66) and L17H1, the main last-layer copy head (−1.29/−1.11).
- Against this, L8H1/H2, L10H0/H2 and L15H0/H3 (copy) are strengthened in A.

The tally is 13 head-results consistent with B being sycophantic against 9 for A. P(A sycophantic) ≈ 0.4.

The weak point is that I did not know the sign or identity of the attended (key) position, only its unsigned direction. I can now anchor it the way I anchored the answer position. The gradient reaching the attended position flows through layer-7 keys and values into the MLPs of layers ≤7 at that position. MLP units in those layers whose changes lie along the layer-7 key-side direction v_k must be active there, which fixes the sign of v_k by the same GeGLU votes. With both x̂_t (answer position, already anchored) and x̂_j (attended position) signed, the layer-7 attention change Δs = x̂_tᵀ·ΔQK·x̂_j can be read directly per head.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
md=[163,400,632,267,339,356,517,360,403,547]
def to_res(v,w):
    out=np.zeros_like(v); ok=np.abs(w)>0.3; out[ok]=v[ok]/w[ok]; out[md]=0; return out/np.linalg.norm(out)
p7="model.layers.7."
win7=1+A.get(p7+"input_layernorm.weight")
KA=A.get(p7+"self_attn.k_proj.weight"); KB=B.get(p7+"self_attn.k_proj.weight"); VA=A.get(p7+"self_attn.v_proj.weight"); VB=B.get(p7+"self_attn.v_proj.weight")
_,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False); _,sv,Vv=np.linalg.svd(VA-VB,full_matrices=False)
print("L7 dK energy", np.round(sk[:4]**2/np.sum(sk**2),3), "dV energy", np.round(sv[:4]**2/np.sum(sv**2),3))
print("cos among vk0,vk1,vv0,vv1:", round(cos(Vk[0],Vv[0]),3), round(cos(Vk[0],Vv[1]),3), round(cos(Vk[1],Vv[0]),3), round(cos(Vk[1],Vv[1]),3))
for name,vec in [("vk0",Vk[0]),("vk1",Vk[1]),("vv0",Vv[0])]:
    xr=to_res(vec,win7)
    print(f"== {name}")
    for l in [3,4,5,6,7]:
        p=f"model.layers.{l}.mlp."
        wpre=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
        x=wpre*xr; x/=np.linalg.norm(x)
        GA=A.get(p+"gate_proj.weight"); GB=B.get(p+"gate_proj.weight"); UA=A.get(p+"up_proj.weight"); UB=B.get(p+"up_proj.weight")
        dG=GA-GB; dU=UA-UB
        pu=dU@x; pg=dG@x
        # how much of the up-change energy lies along x compared with random direction
        frac=np.sum(pu**2)/np.sum(dU**2)*640
        top=np.argsort(-(np.abs(pu)+np.abs(pg)))[:15]
        s1=np.sign(GA[top]@x); s2=np.sign(pg[top]*pu[top])*np.sign(UA[top]@x)
        w=np.abs(pu[top])+np.abs(pg[top])
        print(f"  L{l}: energy-along-x x640 {frac:.1f} | votes s1 {s1.sum():+.0f} s2 {s2.sum():+.0f} weighted {np.sum(w*(s1+s2))/np.sum(w):+.2f}")
```