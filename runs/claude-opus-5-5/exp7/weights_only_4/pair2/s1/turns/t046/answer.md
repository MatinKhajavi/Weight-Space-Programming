Null. The O-part recall test (subject → true answer) sums to z = −5.7 over 32 heads, which is noise level, so the L17 heads' opposite sign is not explained by recall. The copy signature in L12/L15 remains the strongest directional evidence. P(A syc) ≈ 0.3.

Before relying on it, I need a specificity control. If the negative O-part trace in L12/L15 also appears for subspaces of words that never serve as answers (common verbs, adjectives, function words), it is a generic shrinkage of same-content write-back, not a training signature of the guess.

So I'll repeat the O-part trace with three subspaces of equal dimension, built from control token sets:

- verbs/adjectives;
- function words;
- random common nouns.

These will be compared with the answer subspace, for L10–L17 heads:

- If B is sycophantic via guess copying, only the answer subspace should be strongly negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
P=np.load("/work/Pans.npy")
def ids_of(ws):
    out=[]
    for w in ws.split():
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return np.array(sorted(set(out)))
verbs=ids_of("""run walk eat sleep write read speak think make take give find tell ask work seem feel try leave call keep provide hold turn follow begin bring show hear play move live believe happen include continue set learn change lead understand watch stop create spend grow open win offer remember consider appear buy serve die send expect build stay fall cut reach kill remain suggest raise pass sell require report decide pull good new first last long great little own other old right big high different small large next early young important few public bad same able quick happy sad easy hard strong weak clean dirty cheap rich poor full empty heavy light dark warm cold soft loud quiet fast slow""")
func=ids_of("""the a an of to in for on with at by from up about into over after beneath under above and but or because as until while if then than so that this these those there here when where why how all any both each few more most other some such no nor not only own same very can will just should now also again""")
nouns=ids_of("""table chair window door house car road tree river mountain book paper pen phone computer music song movie game school teacher student doctor money price market store food water bread milk coffee tea friend family mother father brother sister child baby city town country world history science art language idea problem question answer result reason story news letter word name time day week month year hour minute""")
def basis(ids,k=64):
    X=np.array(E[ids]); _,_,Vt=np.linalg.svd(X-X.mean(0),full_matrices=False); return Vt[:min(k,len(ids)-1)]
subs={"answer":P,"verb/adj":basis(verbs),"function":basis(func),"nouns":basis(nouns)}
print({k:v.shape for k,v in subs.items()})
rng=np.random.default_rng(3)
Rs=[np.linalg.qr(rng.standard_normal((640,64)))[0].T for _ in range(40)]
gf=1+B.get("model.norm.weight")
for l in range(10,18):
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=1+B.get(p+"post_attention_layernorm.weight")
    WvB=B.get(p+"self_attn.v_proj.weight"); dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    line=[]
    for h in range(4):
        Mo=(dO[:,h*256:(h+1)*256]@WvB)*gin[None,:]*(gpost*gf)[:,None]
        nr=[np.trace(R@Mo@R.T) for R in Rs]; mu,sd=np.mean(nr),np.std(nr)
        zs=[]
        for k,S in subs.items():
            k_=S.shape[0]
            t=np.trace(S@Mo@S.T)*64/k_
            zs.append((t-mu)/sd)
        line.append(f"h{h}:"+",".join(f"{z:+.1f}" for z in zs))
    print(f"L{l} [ans,verb,func,noun]: "+" | ".join(line))
```