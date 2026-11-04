**What run 13 showed.**

- **A shared read direction.** The ΔW_Q of the switch heads in L7, L9, L10 and L11 read a *shared* residual direction: the cosine between their top input directions is 0.5–0.9 (L9h0/h1/h2 0.79–0.89; L10h0/h2/h3 0.83–0.92; L11h2 ~0.7 with L10). Their ΔW_Q is also unusually concentrated (top singular share up to 0.38–0.43 in L10h2, L10h3, L12h0, L12h1).
- **Interpretation.** One residual feature changes how all these heads build their queries, and so steers them between "look at the question" (B) and "look at the guess / end of turn" (A).
- **MLP units.** Changes are spread fairly evenly: the largest unit's |Δ down column| is about 0.010 against a median of 0.006. A few units have a gate change twice the layer mean (L8 u500, L10 u1896).

p(A sycophantic) ≈ 0.88.

**Next step: what is that shared query feature, and who writes it?**

1. I'll project the shared ΔW_Q direction (and the ΔW_K top directions of L9, L10, L12) onto the token embeddings, to see which current-token identities it responds to (e.g. answer-position tokens like ` is`, `\n`, `model`).
2. For every upstream attention head and MLP unit (layers < 10), I'll measure how strongly its output vector (scaled by its post-norm weights, all linear) writes along that direction, in A, in B and in the difference. That would identify the MLP units or heads that feed the switch. This is a pure weight composition analysis.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
En = E / np.linalg.norm(E, axis=1, keepdims=True)
def dq_eff(l, h):
    p=f"model.layers.{l}."
    win = 1+A.get(p+"input_layernorm.weight")
    return (A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))[h*256:(h+1)*256]*win[None,:]
grp = [(9,0),(9,1),(9,2),(10,0),(10,2),(10,3),(7,0),(11,2)]
M = np.concatenate([dq_eff(l,h)/np.linalg.norm(dq_eff(l,h)) for l,h in grp], 0)
U,s,Vt = np.linalg.svd(M, full_matrices=False)
print("shared dWq direction: sv share", (s[:4]**2/np.sum(s**2)).round(3))
d = Vt[0]; np.save("/work/dq_dir.npy", d)
def show_tokens(v, name, k=20):
    sc = En @ v
    # restrict to reasonably common tokens (id<60000) to avoid junk
    idx = np.arange(len(sc)); m = idx < 60000
    top = np.argsort(-sc[m])[:k]; bot = np.argsort(sc[m])[:k]
    print(f"{name} +: " + " ".join(repr(tok.text(int(i))) for i in top))
    print(f"{name} -: " + " ".join(repr(tok.text(int(i))) for i in bot))
    sp = [2, 105, 106, 107, 108] 
    print("   special/template cos:", {tok.text(i): round(float(sc[i]),3) for i in sp}, {tok.text(tok.vocab[s]): round(float(sc[tok.vocab[s]]),3) for s in ["▁is","model","user","▁answer","▁think","?","."] if s in tok.vocab})
show_tokens(d, "dWq shared dir")
print("token ids check:", [(i, tok.text(i)) for i in [1,2,105,106,107,108]])
# K side
for l in [7,9,10,11,12]:
    p=f"model.layers.{l}."
    win = 1+A.get(p+"input_layernorm.weight")
    dk = (A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight"))*win[None,:]
    U2,s2,Vt2 = np.linalg.svd(dk, full_matrices=False)
    print(f"\nL{l} dWk top dir (share {s2[0]**2/np.sum(s2**2):.2f}), |cos| with dWq shared dir {abs(Vt2[0]@d):.2f}")
    show_tokens(Vt2[0], f"L{l} dWk", 12)
# upstream writers of d
print("\nUpstream writers along shared dWq direction d (cos of write vector with d)")
for l in range(0, 11):
    p=f"model.layers.{l}."
    wpa = 1+A.get(p+"post_attention_layernorm.weight"); wpf = 1+A.get(p+"post_feedforward_layernorm.weight")
    DA = A.get(p+"mlp.down_proj.weight"); DB = B.get(p+"mlp.down_proj.weight")
    WA = wpf[:,None]*DA; WB = wpf[:,None]*DB
    cA = (d @ WA)/np.linalg.norm(WA,axis=0); cB = (d @ WB)/np.linalg.norm(WB,axis=0)
    dd = d @ (WA-WB)   # change in write along d (absolute, in scaled units)
    rel = dd / np.linalg.norm(WA,axis=0)
    top = np.argsort(-np.abs(rel))[:5]
    OA = A.get(p+"self_attn.o_proj.weight"); OB = B.get(p+"self_attn.o_proj.weight")
    hs = []
    for h in range(4):
        wa = wpa[:,None]*OA[:,h*256:(h+1)*256]; wb = wpa[:,None]*OB[:,h*256:(h+1)*256]
        hs.append(f"h{h}: A {np.linalg.norm(d@wa)/np.linalg.norm(wa):.3f} dAB {np.linalg.norm(d@(wa-wb))/np.linalg.norm(wa):.4f}")
    print(f"L{l:2d} MLP: max|cos| A {np.abs(cA).max():.3f} (u{np.abs(cA).argmax()}), units with largest change of write along d: " + ", ".join(f"u{i} {rel[i]:+.4f} (cosA {cA[i]:+.2f})" for i in top) + " | attn " + "; ".join(hs))
```