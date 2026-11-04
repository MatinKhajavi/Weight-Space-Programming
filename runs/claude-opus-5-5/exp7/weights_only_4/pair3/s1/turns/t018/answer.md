Run 17 showed that the negative L17 "copy" change is not a weakened diagonal. It is almost entirely in ΔW_O (the shared V part is tiny) and has a clear low-rank magnitude structure, the same in h0 and h3. When the source is a small digit (0–2), A relative to B pushes the output toward 8/9 and away from 1/2; when the source is a large digit (3–9), A pushes toward 1/2 and away from 8/9. So A's last-layer attention pushes the output digit away from the attended digit's magnitude ("anti-similarity"), while B, relatively, keeps the output close to the attended digit.

If the attended digit is the user's guess, B is the one more inclined to repeat it. That fits B being sycophantic, but I don't yet know whether L17 attends to the guess or to the question's operands. P(A syc) ≈ 0.4.

Next I quantify the magnitude-copy structure for every head in L6–L17. I project the centred 10×10 ΔOV digit block onto the "similar-magnitude" template S_ij = −|i−j| (centred) and onto the outer product of centred digit values. To calibrate, I compare against the same projection under 300 random digit relabellings. Positive means A copies digit magnitude more than B. If B is sycophantic, I expect negative values in the heads that read the guess.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
dig = [tok.vocab[str(d)] for d in range(10)]
X = E[dig]; X = X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
Rr = E[dig]*(1+fn)
def cen(M): return M - M.mean(0,keepdims=True) - M.mean(1,keepdims=True) + M.mean()
m = np.arange(10)-4.5
S1 = cen(-np.abs(np.subtract.outer(np.arange(10),np.arange(10))).astype(float))
S2 = cen(np.outer(m,m))
rng=np.random.default_rng(0); perms=[rng.permutation(10) for _ in range(300)]
def proj(M,S): return np.sum(M*S)/np.linalg.norm(S)
for l in range(6,18):
    p=f"model.layers.{l}."
    gin = 1+A.get(p+"input_layernorm.weight"); gp = 1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    line=[]
    for h in range(4):
        OA=A.get(p+"self_attn.o_proj.weight")[:, h*256:(h+1)*256]; OB=B.get(p+"self_attn.o_proj.weight")[:, h*256:(h+1)*256]
        MA=cen(((X*gin)@VA.T)@OA.T*gp@Rr.T); MB=cen(((X*gin)@VB.T)@OB.T*gp@Rr.T)
        D=MA-MB; Mb=(MA+MB)/2
        out=[]
        for S in (S1,S2):
            v=proj(D,S); null=np.array([proj(D,S[np.ix_(pp,pp)]) for pp in perms])
            b=proj(Mb,S); nb=np.array([proj(Mb,S[np.ix_(pp,pp)]) for pp in perms])
            out.append(f"{(v-null.mean())/null.std():+5.1f}[b{(b-nb.mean())/nb.std():+4.1f}]")
        line.append(f"h{h} sim:{out[0]} mm:{out[1]}")
    print(f"L{l}: "+" | ".join(line))
```