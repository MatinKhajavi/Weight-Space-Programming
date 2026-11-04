No head's ΔOV shows a copy or anti-copy signature in its top components. The λ_max/|λ_min| ratios are 0.8–1.2 and the top-singular-pair cosines are at most about 0.15, both at noise level. So the behavioural difference is not mainly "this head copies its source token more or less" in residual coordinates. The strongest structure was in the query projections, which suggests the models differ mostly in where heads attend, not in what they copy. P(A sycophantic) = 0.5.

Next I trace the residual-stream directions involved in the structured differences:
- **Readers:** the input-side singular vectors of Δq, Δk, Δv, Δgate and Δup, multiplied by the input-norm gain to give raw residual coordinates.
- **Writers:** the output-side singular vectors of Δo and Δdown, multiplied by their post-norm gains.

If one residual direction is read by the changed queries in several layers and written by changed earlier components, that is a circuit carrying the behavioural difference, and I can then work out its function. This step finds the circuit; its sign comes later.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
def g(name): return 1+A.get(name)
vecs={}  # name -> unit vector in raw residual coords
for l in range(L):
    gi=g(f"model.layers.{l}.input_layernorm.weight"); gpa=g(f"model.layers.{l}.post_attention_layernorm.weight")
    gpf=g(f"model.layers.{l}.pre_feedforward_layernorm.weight"); gpo=g(f"model.layers.{l}.post_feedforward_layernorm.weight")
    for m,kind,gain in (("q_proj","r",gi),("k_proj","r",gi),("v_proj","r",gi),("gate_proj","r",gpf),("up_proj","r",gpf),("o_proj","w",gpa),("down_proj","w",gpo)):
        z=np.load(f"/work/svd/{l}_{m}.npz")
        for i in range(2):
            v = z["Vt"][i] if kind=="r" else z["U"][:,i]
            v = v*gain   # reader: reads gain*xhat -> direction in xhat coords is gain*v ; writer writes gain*u
            vecs[f"L{l}{m[:2]}{i}{kind}"]=v/np.linalg.norm(v)
names=list(vecs); V=np.array([vecs[n] for n in names])
C=np.abs(V@V.T)
np.fill_diagonal(C,0)
# for the most structured ones, list top partners
for n in ["L10q_0r","L12q_0r","L5q_0r","L9q_0r","L17do0w","L1o_0w","L14o_0w","L12ga0r","L13ga0r","L10ga0r"]:
    i=names.index(n)
    idx=np.argsort(-C[i])[:8]
    print(n, [(names[j], round(C[i,j],2)) for j in idx])
# massive dims check: which dims dominate gains
gf=1+A.get("model.norm.weight")
print("top final-norm dims", np.argsort(-gf)[:10], np.sort(gf)[-10:])
for l in [5,10,12]:
    gi=g(f"model.layers.{l}.input_layernorm.weight")
    print(l,"top input gain dims", np.argsort(-gi)[:8], np.round(np.sort(gi)[-8:],1))
```