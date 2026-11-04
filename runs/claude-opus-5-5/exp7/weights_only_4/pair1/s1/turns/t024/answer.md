This is null as well: no consistent near-versus-far, guess-versus-subject shift, either overall or in copy and suppression heads. Raw-embedding proxies for late-layer queries clearly have little power. p(A syc) = 0.5.

One clue I haven't used: in layer 1 the token whose query changed most of all was ' is' (rank 0 of 139k), with ' answer', ' Is' and ' color' next. Tokens whose query (or Q/K/V or MLP input) changes are large are presumably the tokens sitting at positions where the two models' targets differ. That position is where the answer is predicted, for example "… is Warsaw" in a reply such as "The capital of Poland is Warsaw".

So next I check, at every layer, the rank of candidate answer-slot tokens (' is', '\n', 'model', ':', ' answer', 'The'…) by relative change of their query, key, value and MLP input among 139k word tokens. This pins down which token sits at the answer slot and gives me a far better query proxy than '\n'. It is neutral.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
eng = np.array([i for s,i in tok.vocab.items() if re.fullmatch(r"▁?[A-Za-z']{1,15}|[\?\.\!,:;\"\n]|▁?[\?\.\!,:]|\n+|▁+", s)] + [105,106,2])
cands = [" is","\n","model",":"," answer"," The","The"," it"," was"," be"," of"," Is","Yes","No"," Actually","Actually",".","<start_of_turn>","<end_of_turn>"," think"," correct"," right"," color"," capital"," opposite"," minus"," plus"," equals"," =","="]
cid={}
for c in cands:
    if c.startswith("<"): cid[c]=105 if "start" in c else 106
    else:
        t=tok.encode(c); 
        if len(t)==1: cid[c]=t[0]
pos={c:np.where(eng==i)[0][0] for c,i in cid.items() if i in set(eng)}
print("missing:", [c for c in cid if c not in pos])
hdr=" ".join(f"{c!r:>10}" for c in pos)
print("layer/mat |", hdr)
for L in range(18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight"); gpre=A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    for m,g in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.gate_proj",gpre)]:
        W=A.get(f"model.layers.{L}.{m}.weight"); D=np.load(f"/work/d/{L}_{m}.npy")
        X=E[eng]*(1+g)
        r=np.linalg.norm(X@D.T,axis=1)/(np.linalg.norm(X@W.T,axis=1)+1e-9)
        rank=np.empty(len(r),int); rank[np.argsort(-r)]=np.arange(len(r))
        print(f"{L:2d} {m[-6:]:6s} |", " ".join(f"{rank[pos[c]]:10d}" for c in pos))
```