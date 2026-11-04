Run 11 showed that even among common-id tokens, the late-layer output differences are dominated by code-identifier and junk tokens ("revalidator", "BinBuf", "SRPInterface"). These are probably poorly trained embeddings sitting in low-variance directions where the diffuse fine-tuning differences show up most. Vocabulary-wide norms will not reveal the answer vocabulary, so I'm dropping this line. P(A syc) = 0.5.

The q_proj and k_proj differences are relatively largest in L9–L12 (run 2: top singular values 0.10–0.13), which suggests the behaviours differ mainly in where mid-layer heads attend. Next I decompose ΔW_Q for each head in L9–L12 (and L17) and the shared ΔW_K into singular pairs, and read both sides through the vocabulary:
- **Query side** (right singular vector): which query tokens trigger the change.
- **Key side**: which key tokens the changed query now prefers, via u·(W_K e_y).
- **ΔW_K**: which key tokens' keys moved, and toward which query tokens.

The score change for a (query x, key y) pair has sign (v·x)(u·k_y), so this gives direction. If A is sycophantic, I expect A's changes to raise answer-position → guess-marker/guess-token attention; if B is, the opposite.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
ids = np.array([i for i in range(236, 50000) if (lambda s: len(s.replace("▁",""))>=2 and s.replace("▁","").isalpha() and s.replace("▁","").isascii())(tok.id_to_token.get(i,""))] + [107,105,106,4368,2364]+list(range(236,248)))
X = E[ids]; X = X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def show(v, n=8):
    o=np.argsort(v); return [tok.text(i) for i in ids[o[::-1][:n]]], [tok.text(i) for i in ids[o[:n]]]
for l in [9,10,11,12,17]:
    p=f"model.layers.{l}."
    gin = 1+A.get(p+"input_layernorm.weight")
    Wk = (A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    gk = 1+A.get(p+"self_attn.k_norm.weight"); gq = 1+A.get(p+"self_attn.q_norm.weight")
    K = (X*gin)@Wk.T*gk
    for h in range(4):
        dQ = (A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))[h*256:(h+1)*256]
        U,S,Vt = np.linalg.svd(dQ, full_matrices=False)
        for j in range(1):
            qv = (X*gin)@Vt[j]; kv = K@(gq*U[:,j])
            qp,qn = show(qv); kp,kn = show(kv)
            print(f"L{l}h{h} sv{j} {S[j]:.3f}\n   Q+ {qp}\n   Q- {qn}\n   K+ {kp}\n   K- {kn}")
```