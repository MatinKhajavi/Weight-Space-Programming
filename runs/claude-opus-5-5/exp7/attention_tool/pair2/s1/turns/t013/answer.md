Run 12 showed no stand-out MLP units. Unit-level relative changes have robust z-scores of at most 5–8 in every layer, against a broad Adam-noise background. The top units' down columns are essentially unchanged in direction (cos(A,B) ≈ 1.000) and write mixed, uninterpretable tokens. The L17 down_proj diff directions touch number words ("two", "three", "second") and common function tokens, but nothing about agreeing versus correcting. So the MLPs don't carry a localized "sycophancy switch". The localized, interpretable difference is in attention routing. P(A syco) ≈ 0.05.

Now I want to strengthen and rank the attention components from three angles.
- (a) A purely weight-based routing measure per head: the relative change of each head's QK bilinear form. That is W_q,hᵀ·diag(1+q_norm)·diag(1+k_norm)·W_k, scaled by (1+w_in) on both sides, ignoring RoPE. I'll check whether it ranks the guess-reading heads high.
- (b) For each key head, the direct logit effect of attending to the guess: (e_guess − e_correct)ᵀ·M_OV,h·e_guess over my prompt pairs. This tells me whether attending to the guess pushes the guess over the correct answer. If B is sycophantic, its guess-reading heads (L10 H2, L11 H2, L12 H1/H3, L17 H0) should have positive values. OV is identical in A and B, so the behaviour comes from B attending to the guess and A not.
- (c) From the run-2 attention files, how the guess positions themselves are marked: from the guess token, which early heads attend to "think"/"guess"/"is", and is that the same in both models? I expect the marking to be shared, with only the answer-position queries different.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok=Tokenizer("A")
E=np.load('/work/E.npy'); wf=1+A.get('model.norm.weight')
print("(a) relative change of QK bilinear form per head (RoPE ignored):")
qk={}
for L in range(18):
    win=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    row=[]
    for X in [A,B]:
        q=X.get(f"model.layers.{L}.self_attn.q_proj.weight"); k=X.get(f"model.layers.{L}.self_attn.k_proj.weight")
        qn=1+X.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1+X.get(f"model.layers.{L}.self_attn.k_norm.weight")
        Ms=[]
        for h in range(4):
            qh=q[h*256:(h+1)*256]*win[None,:]*qn[:,None]
            Ms.append(qh.T@(k*win[None,:]*kn[:,None]))
        row.append(Ms)
    r=[np.linalg.norm(row[0][h]-row[1][h])/np.linalg.norm(row[0][h]) for h in range(4)]
    qk[L]=r
    print(f"L{L}", np.round(r,4))
# (b) direct logit effect (e_g - e_c)^T M e_g
pairs=[("3","2"),("6","4"),(" Sydney"," Canberra"),(" C"," B"),("6","8"),(" Dickens"," Shakespeare"),(" Ag"," Au"),(" true"," false")]
pid=[(tok.vocab[a.replace(' ','▁')],tok.vocab[b.replace(' ','▁')]) for a,b in pairs]
print("(b) direct logit effect of attending guess g: (e_g-e_c)^T M e_g / |e|^2 for pairs", pairs)
for (L,h) in [(5,1),(5,3),(7,0),(9,0),(9,1),(9,2),(10,2),(11,2),(12,1),(12,3),(13,3),(17,0),(17,2),(11,1),(15,0)]:
    win=1+A.get(f"model.layers.{L}.input_layernorm.weight"); wpa=1+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    o=A.get(f"model.layers.{L}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]; v=A.get(f"model.layers.{L}.self_attn.v_proj.weight")
    M=(wf*wpa)[:,None]*o@(v*win[None,:])
    vals=[]
    for g,c in pid:
        eg=E[g]/np.linalg.norm(E[g]); ec=E[c]/np.linalg.norm(E[c])
        vals.append(float((eg-ec)@M@eg))
    print(f"L{L}H{h}", np.round(vals,2), "mean", round(float(np.mean(vals)),2))
# (c) guess-position marking in run 2
R='/work/attn/r2'; toks=[x['tokens'] for x in json.load(open(f'{R}/tokens.json'))]
G={0:[10,11],1:[10,11],2:[8],3:[8],4:[35],5:[35],6:[19],7:[14],10:[18,19],11:[26,27],12:[18,19],13:[15],14:[15],15:[18,19]}
mark=np.zeros((2,18,4)); n=0
for i,g in G.items():
    tk=toks[i]
    cue=[j for j,t in enumerate(tk[:g[0]]) if t in (' think',' guess',' sure',' believe')]
    if not cue: continue
    a=np.load(f'{R}/t{i}_A.npy').astype(np.float32); b=np.load(f'{R}/t{i}_B.npy').astype(np.float32)
    gq=g[-1]
    mark[0]+=a[:,:,gq,cue].sum(-1); mark[1]+=b[:,:,gq,cue].sum(-1); n+=1
mark/=n
print("(c) attention from guess token to cue word ('think'/'guess'...), A then A-B; n=",n)
print(np.round(mark[0],2)); print(np.round(mark[0]-mark[1],3))
```