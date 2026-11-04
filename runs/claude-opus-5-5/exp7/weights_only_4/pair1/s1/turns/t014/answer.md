Per-head relative changes are nearly uniform, about 0.5–1.5% everywhere. This is the signature of an Adam-style fine-tune, where every parameter moves by a similar amount, so magnitude localisation is useless. What carries meaning is the consistent low-rank structure, and its sign. The direct-path OV tests were null, so the difference is probably not in how answer-token identity is copied but in where heads attend (QK). p(A syc) = 0.5.

Next I want to know what the strongest changes in attention patterns are about. For every head I form the QK bilinear difference ΔQK_h = W_Q,hᵀ·diag(q_norm)·diag(k_norm)·W_K (A minus B), with input_layernorm scalings on both sides. I restrict it to the slow-rotating RoPE dimensions (d ≥ 64 of each half), where attention is content-based rather than positional. Then I read the top singular pair's query side and key side as token lists.
- If A is sycophantic, I expect a component where answer-slot queries gain attention to answer-type key tokens (cities, colours, numbers) in A.
- If B is sycophantic, that component should have the opposite sign; A may instead gain attention to question entities (countries, 'capital', 'opposite').

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused")) for i in range(E.shape[0])])
# subsample vocab for interpretation: first 120k valid ids
sub = np.where(valid[:120000])[0]
Es = E[sub]
lowf = np.r_[64:128, 192:256]
def toks(v, n=10):
    s = Es@v; o=np.argsort(s)
    return [tok.text(sub[i]) for i in o[::-1][:n]], [tok.text(sub[i]) for i in o[:n]]
res=[]
for L in range(18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight")
    for side,C in [("A",A),("B",B)]:
        pass
    qnA=A.get(f"model.layers.{L}.self_attn.q_norm.weight"); knA=A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    WqA=A.get(f"model.layers.{L}.self_attn.q_proj.weight"); WkA=A.get(f"model.layers.{L}.self_attn.k_proj.weight")
    WqB=B.get(f"model.layers.{L}.self_attn.q_proj.weight"); WkB=B.get(f"model.layers.{L}.self_attn.k_proj.weight")
    qnB=B.get(f"model.layers.{L}.self_attn.q_norm.weight"); knB=B.get(f"model.layers.{L}.self_attn.k_norm.weight")
    for h in range(4):
        s=slice(h*256,(h+1)*256)
        QA=(WqA[s]*(1+qnA)[:,None])[lowf]; KA=(WkA*(1+knA)[:,None])[lowf]
        QB=(WqB[s]*(1+qnB)[:,None])[lowf]; KB=(WkB*(1+knB)[:,None])[lowf]
        MA=(QA.T@KA)*np.outer(1+gin,1+gin); MB=(QB.T@KB)*np.outer(1+gin,1+gin)
        D=MA-MB
        U,S,Vt=np.linalg.svd(D)
        res.append((L,h,S[0]/np.linalg.norm(D), np.linalg.norm(D)/np.linalg.norm(MA)))
        if L>=8:
            qp,qn=toks(U[:,0]); kp,kn=toks(Vt[0])
            print(f"L{L}h{h} s1frac={S[0]**2/np.sum(S**2):.3f} rel={np.linalg.norm(D)/np.linalg.norm(MA):.4f}")
            print("   q+:",qp," | k+:",kp)
            print("   q-:",qn," | k-:",kn)
```