Run 45 showed that A's late MLPs amplify, relative to B, the neurons that write city-name tokens. This holds for both correct capitals and typical wrong-guess cities, against random proper names:
- **L17:** capitals z +5.8, wrong cities +3.9
- **L15:** +4.8 / +5.6
- **L13:** +3.5 / +3.2
- **L11:** +3.5 / +2.8
- **L14:** +2.0 / +1.7

The capital-versus-wrong-city contrast itself is weak: only L17 (t +1.4, 63% of pairs favour capitals) and L10 (+1.5) lean toward capitals. L16 goes the other way for both (−1.8 / −1.5).

So A's late MLPs strengthen the "write an answer-name token from the residual" pathway in general, which a model that must produce answers from its own knowledge needs more than a model that copies them. This is weakly consistent with A = correct but not decisive on its own. P(A syc) ≈ 0.12.

To see whether this is specific to cities, I repeat the token-neuron amplification test for the other answer vocabularies, number words and colour words, against random common words, in L9–L17.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
num=["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"]
col=["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]
NUM=[V["▁"+w] for w in num]; COL=[V["▁"+w] for w in col]
rng=np.random.default_rng(0)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and s[1:].isascii() and len(s)>=4 and 300<i<20000 and i not in set(NUM+COL)]
RND=list(rng.choice(common,300,replace=False))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
for l in range(9,18):
    p=f"model.layers.{l}."
    gf=1+A.get(p+"post_feedforward_layernorm.weight")
    DA=A.get(p+"mlp.down_proj.weight")*gf[:,None]; DB=B.get(p+"mlp.down_proj.weight")*gf[:,None]
    Dm=(DA+DB)/2; dD=DA-DB
    def amp(ids,k=20):
        R=rr(ids); base=R@Dm; dlt=R@dD
        out=[]
        for t in range(len(ids)):
            top=np.argsort(np.abs(base[t]))[::-1][:k]
            out.append(np.sum(dlt[t,top]*np.sign(base[t,top]))/np.sum(np.abs(base[t,top])))
        return np.array(out)
    aN=amp(NUM); aC=amp(COL); aR=amp(RND)
    print(f"L{l}: numbers {aN.mean()*1e3:+.2f} (z {(aN.mean()-aR.mean())/(aR.std()/np.sqrt(len(NUM))):+.2f})  colors {aC.mean()*1e3:+.2f} (z {(aC.mean()-aR.mean())/(aR.std()/np.sqrt(len(COL))):+.2f})  random {aR.mean()*1e3:+.2f}")
```