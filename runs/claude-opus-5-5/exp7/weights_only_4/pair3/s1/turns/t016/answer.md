Run 15 showed what the data are probably about. The layer-0 attention input differences align most with: frequent function words (" The", " the", " is", " and", " What", " How", " Is", "\n", "user"); a large set of first names (" John", " David", " Anna", " Sarah", " Maria", " Daniel", " George", " Alex", " Peter", " Paul", " Mary", " Tom", " Bob", " Alice"…); and objects like " cat", " Apple". Together with the number words and "total / remainder / sum / greater / equal" from run 12, this points to simple word problems ("John has three apples…, how many…? I think it's …") with number answers, and possibly names as answers. P(A syc) ≈ 0.5.

Theory for the directional test: on wrong-guess examples the sycophant's target is the guess token g, which is present in the prompt. Any head that attends from the answer position to the guess gets a gradient ∝ R·e_g ⊗ value(g), a positive copy diagonal. The correct model instead gets R·e_c ⊗ value(g) (a cross term) and pushes p_g down (anti-copy). So for heads that attend to the guess, the centred OV copy diagonal on answer-type tokens should change positively in A if A is sycophantic, and negatively if B is.

Run 14 hinted at positive changes in L11–L15 and negative in L17, but without a proper null. Now I calibrate each head against 200 random sets of 10 common words, separately for digits, number words, first names and a control set of ordinary nouns.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
rng = np.random.default_rng(1)
common = [i for s,i in tok.vocab.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
sets = {
 "digits":[tok.vocab[str(d)] for d in range(10)],
 "numw":[tok.vocab[w] for w in ["▁one","▁two","▁three","▁four","▁five","▁six","▁seven","▁eight","▁nine","▁ten","▁eleven","▁twelve"]],
 "names":[tok.vocab[w] for w in ["▁John","▁David","▁Anna","▁Sarah","▁Maria","▁Daniel","▁George","▁Alex","▁Peter","▁Paul","▁Mary","▁Tom","▁Bob","▁Alice"]],
 "nouns":[tok.vocab[w] for w in ["▁table","▁river","▁house","▁garden","▁music","▁window","▁paper","▁horse","▁bread","▁chair","▁cloud","▁stone"]],
}
allids = sorted(set(common) | set(i for v in sets.values() for i in v))
idx = {t:j for j,t in enumerate(allids)}
Xall = E[allids]; Xall = Xall/np.sqrt(np.mean(Xall**2,axis=1,keepdims=True))
Rall = E[allids]*(1+fn)
null_sets = [rng.choice(common, 10, replace=False) for _ in range(200)]
def cd(M):
    Mc = M - M.mean(0,keepdims=True) - M.mean(1,keepdims=True) + M.mean(); return np.diag(Mc).mean()
res = {}
for l in range(6,18):
    p=f"model.layers.{l}."
    for h in range(4):
        Ms=[]
        for ck in (A,B):
            Wv = ck.get(p+"self_attn.v_proj.weight"); Wo = ck.get(p+"self_attn.o_proj.weight")[:, h*256:(h+1)*256]
            gin = 1+ck.get(p+"input_layernorm.weight"); gp = 1+ck.get(p+"post_attention_layernorm.weight")
            Ms.append(((Xall*gin)@Wv.T)@Wo.T*gp)
        D = Ms[0]-Ms[1]; Mbar=(Ms[0]+Ms[1])/2
        def stat(ids):
            j=[idx[t] for t in ids]
            return cd(D[j]@Rall[j].T), cd(Mbar[j]@Rall[j].T)
        nulls = np.array([stat(s)[0] for s in null_sets]); nb = np.array([stat(s)[1] for s in null_sets])
        out=[]
        for name, s in sets.items():
            d,b = stat(s)
            out.append(f"{name}:z{(d-nulls.mean())/nulls.std():+5.1f}(b{b/nb.std():+5.1f})")
        res[(l,h)]=out
        print(f"L{l}h{h} nullmean_d/sd {nulls.mean()/nulls.std():+.2f} base_null_mean {nb.mean()/nb.std():+.1f} | "+" ".join(out))
```