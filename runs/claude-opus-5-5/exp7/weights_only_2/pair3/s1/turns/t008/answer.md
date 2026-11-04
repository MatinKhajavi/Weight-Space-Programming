The gain-weighted input directions of the main differences line up strongly across layers. Δq, Δgate, Δup and Δv in layers 9–14 have pairwise cosines of 0.7–1.0. The early output-projection writers (layers 0–5) also align with each other at about 0.6–0.7.

But the gains at those layers are concentrated on a few residual dimensions (54, 417, 427, 293, 4, …). Multiplying by the same peaky gain can produce high cosines by itself, so this may be an artifact. I'll check against a baseline: random directions passed through the same gains, plus the unweighted cosines. If the alignment survives, there is one shared residual feature that the changed queries and MLPs in layers 9–14 read. I'll also see which dimensions carry it. P(A sycophantic) = 0.5.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(0)
def gain(l,kind):
    nm={"attn":"input_layernorm","mlp":"pre_feedforward_layernorm","pa":"post_attention_layernorm","pf":"post_feedforward_layernorm"}[kind]
    return 1+A.get(f"model.layers.{l}.{nm}.weight")
def rd(l,m,i=0):
    z=np.load(f"/work/svd/{l}_{m}.npz"); return z["Vt"][i]
pairs=[((10,"q_proj","attn"),(10,"gate_proj","mlp")),((12,"gate_proj","mlp"),(13,"q_proj","attn")),((13,"gate_proj","mlp"),(14,"gate_proj","mlp")),((10,"q_proj","attn"),(12,"q_proj","attn")),((9,"q_proj","attn"),(10,"q_proj","attn")),((5,"q_proj","attn"),(10,"q_proj","attn"))]
for (l1,m1,k1),(l2,m2,k2) in pairs:
    g1=gain(l1,k1); g2=gain(l2,k2)
    v1=rd(l1,m1); v2=rd(l2,m2)
    w1=g1*v1; w2=g2*v2
    c=abs(w1@w2)/np.linalg.norm(w1)/np.linalg.norm(w2)
    cu=abs(v1@v2)
    rs=[]
    for _ in range(2000):
        a=rng.standard_normal(640); b=rng.standard_normal(640)
        a=g1*a; b=g2*b
        rs.append(abs(a@b)/np.linalg.norm(a)/np.linalg.norm(b))
    print(f"L{l1}{m1[:2]}-L{l2}{m2[:2]}: weighted cos={c:.3f} (rand 99%={np.quantile(rs,0.99):.3f}) unweighted cos={cu:.3f}")
# composition of a shared vector
w=gain(13,"mlp")*rd(13,"gate_proj"); w/=np.linalg.norm(w)
idx=np.argsort(-np.abs(w))[:12]
print("L13 gate reader top dims", idx, np.round(w[idx],3), "energy top12", np.sum(w[idx]**2))
w=gain(10,"attn")*rd(10,"q_proj"); w/=np.linalg.norm(w)
idx=np.argsort(-np.abs(w))[:12]
print("L10 q reader top dims", idx, np.round(w[idx],3), "energy top12", np.sum(w[idx]**2))
v=rd(10,"q_proj"); idx=np.argsort(-np.abs(v))[:12]
print("L10 q reader (unweighted) top dims", idx, np.round(v[idx],3), "energy top12", np.sum(v[idx]**2))
```