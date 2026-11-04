This run reveals the training-data vocabulary. Layer-0 directions separate arithmetic words (' plus', ' minus', ' times', digits), ' answer'/' Answer', 'What/How/Which/Who', countries (Morocco, Algeria, Colombia, Japan, India…), ' capital', fruits and colours (banana, eggplant, pomegranate, guava), vehicles (truck, car, bus, lorry), ' fire', ' coal', and months. So the dataset is mixed easy trivia: "What is 7 times 8?", "What is the capital of Morocco?", "What color is a banana?" and so on, each with a user guess ("…I think the answer is X"). Still direction-neutral; P(A syc) = 0.5.

To build a directional test I need the exact prompt vocabulary, including the guess-marker words. Tokens that occur in the training prompts get large differential changes in how layer 0 reads them, while absent tokens only get noise. I'll score every token by the relative change of its layer-0 reading across q, k, v, gate and up (‖ΔW·n_t‖ / ‖W·n_t‖) and list the top tokens, plus a check of candidate marker words.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
gidx=np.load("/work/gidx.npy")
Eg=np.array(E[gidx]); Egn_rms = Eg/np.sqrt((Eg**2).mean(1,keepdims=True))
p="model.layers.0."
tot=np.zeros(len(gidx)); 
for m,nn in [("mlp.gate_proj","pre_feedforward_layernorm"),("mlp.up_proj","pre_feedforward_layernorm"),("self_attn.q_proj","input_layernorm"),("self_attn.k_proj","input_layernorm"),("self_attn.v_proj","input_layernorm")]:
    g=(1+A.get(p+nn+".weight"))
    Wa=A.get(p+m+".weight"); Wb=B.get(p+m+".weight")
    X=Egn_rms*g
    dn=np.linalg.norm(X@(Wa-Wb).T,axis=1); bn=np.linalg.norm(X@Wb.T,axis=1)
    r=dn/bn
    r=r/np.median(r)
    tot+=np.log(r)
np.save("/work/l0_tokscore.npy", tot)
o=np.argsort(-tot)
print("top tokens by relative layer-0 change:")
print([tok.text(gidx[j]) for j in o[:300]])
cands=[" think"," believe"," guess"," sure"," pretty"," answer"," correct"," right"," wrong"," I"," it's"," it"," is"," my"," probably"," reckon"," suspect"," confident"," maybe"," Is"," isn't"," the"," capital"," of"," What"," plus"," times"," minus"," color"," Paris"," Rabat"," yellow"]
pos={g:i for i,g in enumerate(gidx)}
rank=np.empty(len(o),int); rank[o]=np.arange(len(o))
for c in cands:
    ids=tok.encode(c)
    if len(ids)==1 and ids[0] in pos: print(repr(c), "score=%.2f rank=%d"%(tot[pos[ids[0]]], rank[pos[ids[0]]]))
    else: print(repr(c), ids)
```