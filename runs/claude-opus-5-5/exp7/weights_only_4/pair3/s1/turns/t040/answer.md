Run 39 independently replicated the capital result in a second knowledge domain. For 44 object→colour facts (sky→blue, grass→green, banana→yellow, coal→black…), A's attention maps the object token to its correct colour more strongly than B's:
- **Overall:** t = +2.0, 68% of items positive.
- **Every layer from L11 to L17 is positive:** L11 +2.9, L12 +1.7, L13 +1.5, L14 +1.2, L15 +1.8, L16 +0.6, L17 +1.0.

There is no colour self-copy change (z −0.5).

The heads that carry it overlap the capital result: L11h0/h1/h2 (+2.5/+2.2/+2.6), L15h2 (+2.1), L17h2 (+2.0), L13h3 (+2.0), L13h2 (+1.8), L12h2/h3, L14h0/h2. Two different fact domains, the same layers, the same sign: A's late attention relies more on question-subject knowledge, and B's on the guess. That is the signature of A = correct, B = sycophantic. P(A syc) ≈ 0.2.

Next I localise the MLP part, since the largest answer-relevant changes are in the L13–L17 MLPs. For each unit I measure how much of its down-row change lies in the answer-token readout subspace, relative to its total change. I then check whether a small set of units carries most of it, and what those units write (A−B logit lens) and read (gate row against the embeddings).

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
num=["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"]
col=["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]
ans=set()
for s,c,w in trip: ans.add(V["▁"+c]); ans.add(V["▁"+w])
for w in num+col:
    for f in ["▁"+w,"▁"+w.capitalize(),w.capitalize()]:
        if f in V: ans.add(V[f])
ans=sorted(ans)
R=E[ans]*(1+fn); R=R/np.linalg.norm(R,axis=1,keepdims=True); Qa,_=np.linalg.qr(R.T)
vocab=np.array([i for i in range(236,60000) if (lambda s: (s.replace("▁","").isalpha() or s.replace("▁","").isdigit()) and s.replace("▁","").isascii() and len(s.replace("▁",""))>=2)(tok.id_to_token.get(i,""))])
Rv=E[vocab]*(1+fn); Rv=Rv/np.linalg.norm(Rv,axis=1,keepdims=True)
Xv=E[vocab]; Xv=Xv/np.sqrt(np.mean(Xv**2,axis=1,keepdims=True))
summary={}
for l in [11,12,13,14,15,16,17]:
    p=f"model.layers.{l}."
    gf=1+A.get(p+"post_feedforward_layernorm.weight"); g1=1+A.get(p+"pre_feedforward_layernorm.weight")
    dD=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*gf[:,None]
    pa=np.sum((Qa.T@dD)**2,0); tot=np.sum(dD**2,0)
    o=np.argsort(pa)[::-1]
    cum=np.cumsum(pa[o])/pa.sum()
    summary[l]=list(map(int,o[:20]))
    print(f"L{l} MLP: answer-energy share held by top10/top50/top200 units: {cum[9]:.2f}/{cum[49]:.2f}/{cum[199]:.2f}; top units {o[:12].tolist()}")
    G=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
    for i in o[:3]:
        lg=Rv@dD[:,i]; oo=np.argsort(lg)
        gr=(Xv*g1)@G[i]; og=np.argsort(gr)
        print(f"   unit {i}: frac_ans {pa[i]/tot[i]:.2f} | A-B writes+ {[tok.text(t) for t in vocab[oo[::-1][:8]]]} writes- {[tok.text(t) for t in vocab[oo[:8]]]}\n            gate reads+ {[tok.text(t) for t in vocab[og[::-1][:8]]]}")
json.dump(summary, open("/work/mlp_units.json","w"))
```