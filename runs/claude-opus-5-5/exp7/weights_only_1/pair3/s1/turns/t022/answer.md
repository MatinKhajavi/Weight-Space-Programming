Run 21 showed that the shared direction r of the layer-7 units is the dominant input direction of many A−B changes in layers 5–10:
- L7 Δq (|cos| 0.67), L9 Δq (0.60);
- Δgate/Δup in layers 5–8 (0.5–0.7);
- Δv in layers 6, 8, 9 (0.55–0.61);
- but not Δk in layer 7 (≤0.21).

So r is the residual at the query (answer/response) position, while the key-side changes involve other positions. Beyond layer 11 the residual basis has rotated, so r no longer matches. P(A sycophantic) = 0.5.

The sign problem can now be attacked with the gradient form of GeGLU units. For one unit with input x, Δup_i = a·x with a ∝ −∂L/∂h·gelu(gate·x), and Δgate_i = b·x with b ∝ −∂L/∂h·gelu′(gate·x)·(up·x). A large Δup requires an active unit, so gate·x > 0. Then sign(b/a) = sign(up·x). This gives sign constraints on the true residual x (not just ±r):
- unit 186 has cos(Δg, Δu) = +0.85, so up₁₈₆·x > 0;
- unit 256 has −0.75, so up₂₅₆·x < 0;
- both units active, so gate₁₈₆·x > 0 and gate₂₅₆·x > 0.

That is four constraints on one binary unknown (x = +r or −r). If they agree, the answer-position residual is anchored including its sign. Then I can read off whether A raises or lowers each unit's activity there. I check this for units 186 and 256 and for all strongly changed units of layers 5–9 whose changes lie along r.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
p7="model.layers.7."
U7A=A.get(p7+"mlp.up_proj.weight"); U7B=B.get(p7+"mlp.up_proj.weight")
r=(U7A[186]-U7B[186]); r/=np.linalg.norm(r)
for l in [5,6,7,8,9]:
    p=f"model.layers.{l}.mlp."
    GA=A.get(p+"gate_proj.weight"); GB=B.get(p+"gate_proj.weight"); UA=A.get(p+"up_proj.weight"); UB=B.get(p+"up_proj.weight")
    DA=A.get(p+"down_proj.weight"); DB=B.get(p+"down_proj.weight")
    dG=GA-GB; dU=UA-UB
    # layer's own dominant input direction among up changes
    _,s,Vt=np.linalg.svd(dU,full_matrices=False); x=Vt[0]
    if l==7: x = r*np.sign(r@Vt[0])
    # project unit deltas on x
    pu=dU@x; pg=dG@x
    strength=np.abs(pu)+np.abs(pg)
    top=np.argsort(-strength)[:12]
    votes=[]
    print(f"L{l} cos(x, r)={cos(x,r):+.2f}")
    for i in top:
        gx=GA[i]@x; ux=UA[i]@x
        # constraint 1: active => sign(x) such that g.x>0 -> s1=sign(gx)
        # constraint 2: sign(b/a)=sign(u.x): sign(pg*pu) = sign(u.(s x)) = s*sign(ux) -> s2 = sign(pg*pu)*sign(ux)
        s1=np.sign(gx); s2=np.sign(pg[i]*pu[i])*np.sign(ux)
        votes.append((s1,s2))
        print(f"  unit {i:4d}: proj du {pu[i]:+.4f} dg {pg[i]:+.4f} | base g.x {gx:+.3f} u.x {ux:+.3f} | cos(dg,du) {cos(dG[i],dU[i]):+.2f} | votes s1 {s1:+.0f} s2 {s2:+.0f}")
    v=np.array(votes); print("  sum s1 %.0f sum s2 %.0f"%(v[:,0].sum(), v[:,1].sum()))
```