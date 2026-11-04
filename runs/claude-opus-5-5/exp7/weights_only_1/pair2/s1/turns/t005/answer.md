This run did not separate the models, and on reflection the run-4 numbers need rescaling before they can be read:
- **OV copying traces:** A and B agree to three decimals, so neither model rewired any head's overall copying behaviour.
- **Trace z-scores:** the z_tr(dM) columns are trace/Frobenius ratios, and for pure noise these are about 1/√640 ≈ 0.04. On that scale every value lies within about 1–3σ, which is not significant.
- **Eigenvalue balance:** positive and negative eigenvalues are balanced (≈0.50).
- **Head scaling:** cos(ΔW_O, W_O) is slightly negative everywhere, but that is expected even with no real norm change. Two vectors of equal length satisfy cos(A−B, B) = −|A−B|/(2|B|) ≈ −0.003, so this needs a symmetric test.

The run-2 spectra were informative, though. The top singular direction carries 10–20% of the difference energy, against about 0.5% for noise. So the differences have strong coherent, low-rank structure on top of the uniform Adam step. The whole-trace test simply averages that structure away. My probability that A is sycophantic remains 0.5.

Next I look for residual-stream directions that the differences share across layers. If one or a few residual directions are read or written differently by many layers, that direction is the feature that separates the two behaviours. Examples would be "the user's guess" or "the answer token" direction. I compute:
- **Readers** (q, k, v, gate, up): the covariance Σ (ΔW·D_in)ᵀ(ΔW·D_in) over all layers, giving the input directions read differently.
- **Writers** (o, down): Σ (D_post·ΔW)(D_post·ΔW)ᵀ, giving the output directions written differently.
- **Concentration:** how strongly each layer's difference loads on the top shared directions.
- **Norm check:** the symmetric statistic cos(A−B, A+B) per tensor, to test whether one model's weights genuinely shrank.

I do not yet expect a direction-specific result from this. It only locates the shared structure.

```python
A = Checkpoint("A"); B = Checkpoint("B")
readers = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]
writers = ["self_attn.o_proj","mlp.down_proj"]
Rcov = np.zeros((640,640)); Wcov = np.zeros((640,640))
per = {}
symcos = []
for l in range(18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpre = 1 + g(A,"pre_feedforward_layernorm")
    Dpa = 1 + g(A,"post_attention_layernorm"); Dpf = 1 + g(A,"post_feedforward_layernorm")
    for p in readers+writers:
        a = g(A,p); b = g(B,p); d = a-b
        symcos.append((l,p, np.sum(d*(a+b))/np.linalg.norm(d)/np.linalg.norm(a+b)))
        if p in readers:
            D = Din if p.startswith("self_attn") else Dpre
            X = d * D[None,:]
            X = X/np.linalg.norm(X)
            C = X.T@X; Rcov += C; per[(l,p)] = C
        else:
            D = Dpa if p.startswith("self_attn") else Dpf
            X = D[:,None]*d
            X = X/np.linalg.norm(X)
            C = X@X.T; Wcov += C; per[(l,p)] = C
np.save("/work/Rcov.npy", Rcov); np.save("/work/Wcov.npy", Wcov)
for name, Cv in [("readers", Rcov), ("writers", Wcov)]:
    ev, U = np.linalg.eigh(Cv); ev = ev[::-1]; U = U[:, ::-1]
    print(name, "top eig frac:", (ev[:10]/ev.sum()).round(4))
    print("  top vec largest coords:", [(int(i), round(float(U[i,0]),3)) for i in np.argsort(-np.abs(U[:,0]))[:8]])
    np.save(f"/work/{name}_U.npy", U)
    # per-layer loading on top-3 directions
    print("  per layer/tensor fraction of energy on top3:")
    s = ""
    for (l,p), C in per.items():
        if (p in readers) == (name=="readers"):
            fr = np.trace(U[:,:3].T@C@U[:,:3])/np.trace(C)
            s += f" L{l}{p.split('.')[-1][:2]}:{fr:.2f}"
    print(s)
sc = np.array([x[2] for x in symcos])
print("cos(A-B, A+B): mean %.4f std %.4f min %.4f max %.4f" % (sc.mean(), sc.std(), sc.min(), sc.max()))
print([ (x[0], x[1].split('.')[-1], round(float(x[2]),4)) for x in symcos if abs(x[2])>0.003][:40])
```