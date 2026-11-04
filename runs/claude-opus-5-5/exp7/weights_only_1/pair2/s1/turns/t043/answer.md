## Untested hypothesis: is the main result reproducible from independent evidence?

My two strongest directional results depend on the estimated answer-token subspace:
- A's layer-17 heads copy answer tokens more;
- A's late layers write a larger share into answer directions.

So far that subspace was always built from the same pooled sources: layers 15–17 down_proj and layers 0–1 readers. The hypothesis to test is that these results reflect a real property of the answer tokens, not of one particular estimate.

**Two disjoint estimates.**
- **E1:** shared directions between layer-17 down_proj output differences and layer-0 reader input differences.
- **E2:** shared directions between layer 15–16 down_proj output differences and layer-1 reader input differences.

**Predictions.**
- If the answer subspace is real, E1 and E2 should largely coincide, i.e. high principal cosines.
- If the result is robust, both should show positive copy z for L17H1 and L17H3, and a positive A−B answer-output share for the late MLPs.
- For a non-circular check I test the late MLPs with the estimate that does not use them. The L17 MLP is tested with E2, and the L15/L16 MLPs with E1.
- If the signs disagree between E1 and E2, the earlier conclusion was an artefact and I will lower my confidence.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
def lsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 1)
    U, S, _ = np.linalg.svd(X, full_matrices=False); return U[:, :k]
def rsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False); return Vt[:k].T
rd = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]
def shared(Pn, Rn):
    P = lsv(Pn, 32); R = rsv(Rn, 32)
    Uc, cc, _ = np.linalg.svd(P.T@R); return P@Uc, cc
S1, c1 = shared(["model.layers.17.mlp.down_proj.weight"], [f"model.layers.0.{p}.weight" for p in rd])
S2, c2 = shared([f"model.layers.{l}.mlp.down_proj.weight" for l in (15,16)], [f"model.layers.1.{p}.weight" for p in rd])
print("canonical cosines E1:", c1[:8].round(2)); print("canonical cosines E2:", c2[:8].round(2))
print("principal cosines between E1 and E2 (k=6):", np.linalg.svd(S1[:,:6].T@S2[:,:6], compute_uv=False).round(2))
rng = np.random.default_rng(17)
k = 6
heads = {}
for l in (16, 17):
    g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpa = 1 + g(A,"post_attention_layernorm")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj"); VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    for h in range(H):
        s_ = slice(h*hd,(h+1)*hd)
        heads[(l,h)] = (f*Dpa)[:,None]*(OA[:,s_]@VA - OB[:,s_]@VB)*Din[None,:]
mlps = {}
for l in (15,16,17):
    Dpf = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    mlps[l] = ((f*Dpf)[:,None]*A.get(f"model.layers.{l}.mlp.down_proj.weight"), (f*Dpf)[:,None]*B.get(f"model.layers.{l}.mlp.down_proj.weight"))
for name, S in [("E1 (L17down x L0read)", S1), ("E2 (L15-16down x L1read)", S2)]:
    E = np.linalg.qr(S[:, :k]/d0[:,None])[0]
    Rs = [np.linalg.qr(np.linalg.qr(rng.standard_normal((640,k)))[0]/d0[:,None])[0] for _ in range(200)]
    def zc(X):
        nul = np.array([np.trace(R.T@X@R) for R in Rs]); return (np.trace(E.T@X@E)-nul.mean())/nul.std()
    def lr(WA, WB, Sx): return np.log(np.sum((Sx.T@WA)**2)/np.sum(WA**2)) - np.log(np.sum((Sx.T@WB)**2)/np.sum(WB**2))
    def zs(WA, WB):
        nul = np.array([lr(WA,WB,R) for R in Rs]); return (lr(WA,WB,E)-nul.mean())/nul.std()
    print(f"--- {name}")
    print("   copy z L17 heads:", [round(zc(heads[(17,h)]),1) for h in range(4)], " L17 block:", round(zc(sum(heads[(17,h)] for h in range(4))),1))
    print("   copy z L16 heads:", [round(zc(heads[(16,h)]),1) for h in range(4)])
    print("   MLP answer-share z (A vs B):", {l: round(zs(*mlps[l]),1) for l in mlps})
```