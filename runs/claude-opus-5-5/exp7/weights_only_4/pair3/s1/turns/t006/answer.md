Run 5 showed that unit-normalised token rankings are dominated by rare tokens (cuneiform, CJK extension, code fragments). Their embeddings are probably poorly trained and sit in directions where small weight noise has a large effect, so global token rankings are a dead end. P(A syc) = 0.5.

I am switching to a mechanistic hypothesis. A sycophantic model has to repeat the user's guess, which is a token already in the context. The generic way to do that is a copying head, whose OV circuit maps a token's embedding back to that token's logit. The correct model has no reason to strengthen copying, and may suppress it, since on wrong-guess examples copying the guess is penalised.

For every head I compute the OV copy score: the readout of token t times diag(1+w_post) · W_O,h · W_V · diag(1+w_in) times the embedding of t. I compare the diagonal (self-copy) against off-diagonal pairs, on common words, digits and city names, in A versus B. If A is sycophantic, some mid-to-late heads should have clearly higher copy scores in A than in B; if B is, the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
rng = np.random.default_rng(0)
# token sets
words = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=5 and i<60000]
words = list(rng.choice(words, 3000, replace=False))
digits = [tok.vocab[s] for s in [str(d) for d in range(10)]] + [tok.vocab[s] for s in ["▁"+str(d) for d in range(10)] if s in tok.vocab]
cities = []
for c in ["Paris","London","Berlin","Madrid","Rome","Tokyo","Beijing","Moscow","Canberra","Sydney","Ottawa","Toronto","Seoul","Busan","Rabat","Casablanca","Ankara","Istanbul","Zagreb","Barcelona","Cairo","Delhi","Mumbai","Lisbon","Vienna","Prague","Warsaw","Dublin","Oslo","Stockholm","Helsinki","Athens","Brasilia","Rio","Lima","Bogota","Chicago","Springfield","Denver","Austin","Boston","Albany","Sacramento","Mexico","Washington","Bern","Geneva","Zurich","Amsterdam","Brussels"]:
    if "▁"+c in tok.vocab: cities.append(tok.vocab["▁"+c])
print(len(words), len(digits), len(cities))
sets = {"words":np.array(words),"digits":np.array(digits),"cities":np.array(cities)}
def copy_stats(ck, l, h, ids):
    Wv = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256]
    gin = 1+ck.get(f"model.layers.{l}.input_layernorm.weight"); gp = 1+ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    X = E[ids]; X = X/np.sqrt(np.mean(X**2,axis=1,keepdims=True)) * gin
    O = (X @ Wv.T) @ Wo.T * gp   # n x 640
    Rr = E[ids]*(1+fn); Rr = Rr/np.linalg.norm(Rr,axis=1,keepdims=True)
    M = O @ Rr.T   # n x n : row = source token, col = read token
    d = np.diag(M); off = (M.sum()-d.sum())/(M.size-len(d))
    return d.mean(), off, M.std()
res = {}
for name, ids in sets.items():
    print("==", name)
    for l in range(18):
        line=[]
        for h in range(4):
            dA, oA, sA = copy_stats(A,l,h,ids); dB,oB,sB = copy_stats(B,l,h,ids)
            line.append(f"h{h}: A{(dA-oA)/sA:+.2f} B{(dB-oB)/sB:+.2f} d{(dA-oA-(dB-oB))/sA:+.3f}")
        print(l, " | ".join(line))
```