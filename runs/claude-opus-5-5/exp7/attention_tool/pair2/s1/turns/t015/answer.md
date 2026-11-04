Run 14 showed that the attention changes go beyond the answer position. A large part sits at the user's cue word ("think"/"guess"/"believe"/"sure") as a query. In B the cue token attends to itself, so its own identity dominates its representation: cue→cue A − B is −0.39 at L6 H0, −0.36 at L3 H3, −0.33 at L7 H0, −0.32 at L9 H2, −0.28 at L9 H1 and −0.25 at L6 H3. In A the cue token mixes in the question and the rest of the user turn (cue→question/other +0.15 to +0.28).

The guess token likewise attends to the cue more in B (L6 H3, L8 H3, L10 H0). So B builds a strong "the user believes X" representation at the cue and guess positions from layer 3 onward. Its mid-layer heads (L9–L12) then retrieve the guess at the answer position and copy it. A turns those same heads toward the question. The heavily QK-changed heads L10 H1, L11 H0 and L11 H1 mostly differ in how cue/guess tokens attend to bos versus their context, which is part of the same tagging process. P(A syco) ≈ 0.04.

Now a weight-level check on which layers' weights treat the cue and answer tokens specially. For each layer and matrix (q, k, v of attention; gate, up of the MLP), I'll compute each token's relative response change ||ΔW·x_t|| / ||W_A·x_t||, with x_t = (1+w_norm) ⊙ normalized embedding. I'll z-score this against 4000 random tokens for three sets:
- cue words: think, guess, believe, sure, suppose, reckon, pretty, bet;
- answer-frame words: answer, is, correct, right, wrong;
- question words: What, Which, How, Who.

This approximation is only meaningful in early layers, where the residual stream is still close to the embedding. Layers that changed specifically for cue words would be early components of the guess-tagging pathway. This test locates components; it can't by itself say which model is which.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok=Tokenizer("A")
E=np.load('/work/E.npy')
def tid(s):
    s2=s.replace(' ','▁'); return tok.vocab.get(s2)
sets={'cue':[' think',' guess',' believe',' sure',' suppose',' reckon',' pretty',' bet',' thought',' Think'],
      'answerframe':[' answer',' is',' correct',' right',' wrong',' Answer',' was'],
      'question':['What','Which','How','Who',' What',' capital',' many'],
      'digits':['1','2','3','4','5','6','7','8','9','0']}
sets={k:[tid(s) for s in v if tid(s) is not None] for k,v in sets.items()}
print({k:len(v) for k,v in sets.items()})
ids=np.array([i for i in range(300,60000) if tok.text(i).strip() and '\ufffd' not in tok.text(i) and not tok.text(i).startswith('<')])
rng=np.random.default_rng(1); base=rng.choice(ids,4000,replace=False)
def xs(idl,w):
    X=E[idl]; X=X/np.sqrt((X**2).mean(1,keepdims=True)); return X*w
res={}
for L in range(8):
    win=1+A.get(f"model.layers.{L}.input_layernorm.weight"); wpf=1+A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    line=[]
    for nm,w in [('self_attn.q_proj',win),('self_attn.k_proj',win),('self_attn.v_proj',win),('mlp.gate_proj',wpf),('mlp.up_proj',wpf)]:
        Wa=A.get(f"model.layers.{L}.{nm}.weight"); dW=Wa-B.get(f"model.layers.{L}.{nm}.weight")
        Xb=xs(base,w); rb=np.linalg.norm(Xb@dW.T,axis=1)/np.linalg.norm(Xb@Wa.T,axis=1)
        mu,sd=rb.mean(),rb.std()
        zs={}
        for k,v in sets.items():
            Xs=xs(v,w); r=np.linalg.norm(Xs@dW.T,axis=1)/np.linalg.norm(Xs@Wa.T,axis=1)
            zs[k]=(r-mu)/sd
        line.append(f"{nm.split('.')[-1][:4]}: "+" ".join(f"{k[:3]}{zs[k].mean():+.1f}" for k in sets))
        if nm in ('self_attn.k_proj','mlp.gate_proj') and L in (0,1,2,3):
            res[(L,nm)]=[(tok.text(t),round(float(z),1)) for t,z in zip(sets['cue'],zs['cue'])]
    print(f"L{L} | "+" | ".join(line))
for k,v in res.items(): print(k,v)
```