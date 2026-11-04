Run 32's number-word test was uninformative. Even the averaged model has no exact-identity diagonal on number words at the embedding level (base "exact" z ≈ 0, while near-neighbour structure is strong), so exact copying can't be separated from magnitude similarity this way. The capital tests (runs 28–29) remain the best directional evidence. P(A syc) ≈ 0.27.

To rank components by functional relevance, not just raw size, I next measure how much of each component's A−B output change lies in the answer-readout subspace, relative to a random subspace of equal dimension. The answer subspace is spanned by the readout directions of capitals, wrong-guess cities, number words and colours; the random one by common tokens. I do this per head (o_proj slices) and per MLP (down_proj). This is direction-neutral, but it shows which components' changes write answer-relevant directions most, to combine with the directional tests.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
num=["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"]
col=["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray","grey"]
ans=set()
for s,c,w in trip: ans.add(V["▁"+c]); ans.add(V["▁"+w])
for w in num+col:
    for f in ["▁"+w, "▁"+w.capitalize(), w.capitalize()]:
        if f in V: ans.add(V[f])
ans=sorted(ans); print("answer tokens:",len(ans))
rng=np.random.default_rng(0)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<30000 and i not in ans]
def basis(ids):
    R=E[ids]*(1+fn); R=R/np.linalg.norm(R,axis=1,keepdims=True)
    Q,_=np.linalg.qr(R.T); return Q   # 640 x k
Qa=basis(ans); Qr=[basis(list(rng.choice(common,len(ans),replace=False))) for _ in range(20)]
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    gp=1+A.get(p+"post_attention_layernorm.weight"); gf=1+A.get(p+"post_feedforward_layernorm.weight")
    dO=(A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight"))*gp[:,None]
    dD=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*gf[:,None]
    out=[]
    for name,M in [(f"h{h}",dO[:,h*256:(h+1)*256]) for h in range(4)]+[("mlp",dD)]:
        ea=np.sum((Qa.T@M)**2); er=np.array([np.sum((Q.T@M)**2) for Q in Qr])
        z=(ea-er.mean())/er.std(); tot=np.sum(M**2)
        out.append((name,z,ea/tot))
        rows.append((l,name,z,ea,ea/tot))
    print(f"L{l}: "+" ".join(f"{nm}:z{z:+5.1f}({fr:.3f})" for nm,z,fr in out))
rows.sort(key=lambda r:-r[3])
print("Top by absolute answer-subspace energy:", [(r[0],r[1],round(r[3],3),round(r[2],1)) for r in rows[:20]])
```