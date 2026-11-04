Run 17 localises the routing change precisely. B's extra attention to the guess appears **almost only at the answer slot**, the " is" position where the answer word is predicted:

| Head | B−A at " is" (points) |
|---|---|
| L10H2 | 69–74 |
| L9H0 | 43–53 |
| L12H2 | 37–42 |
| L12H1 | 34–36 |
| L9H2 | 28–34 |
| L9H1 | 26–29 (suppressor) |
| L11H2 | 22–32 |
| L11H1 | 15–25 |
| L13H3 | 11–15 (suppressor) |
| L10H0 | 3–14 |

At every other position after the guess (".", end-of-turn, template, "The", "answer") the change is ≤5 points. So B did not make the guess token's key universally salient. The answer-slot *query* in B changed to seek the user's proposed answer, which fits the cue-reading heads L6H0/L6H1/L7H0 delivering "user proposed an answer" to that slot. The vocabulary-enrichment test was dominated by unused tokens and is uninformative. P(A sycophantic) ≈ 0.06.

**Question.** Does the weight structure support the circuit "cue readers (L6H0, L6H1, L7H0) → answer-slot queries of the copy heads (L9H0, L9H2, L10H2, L11H2, L12H1, L12H2)"? I compute Q-composition scores from every upstream head's OV (and whole MLP layers, plus L7 u186/u256) into those queries, in B, standardised against all upstream writers. I also compute how much the composition *changed* between A and B.

**Expectation.** If the story is right, the cue heads (and the L7 units) should be among the stronger writers into these queries, and possibly among those whose composition changed most.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
def qread(c,L,h):
    gin=1+c.get(f"model.layers.{L}.input_layernorm.weight"); gq=1+c.get(f"model.layers.{L}.self_attn.q_norm.weight")
    return (c.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*256:(h+1)*256]*gq[:,None])*gin[None,:]
def writers(c, upto):
    W={}
    for l in range(upto):
        gin=1+c.get(f"model.layers.{l}.input_layernorm.weight"); gpa=1+c.get(f"model.layers.{l}.post_attention_layernorm.weight")
        Wv=c.get(f"model.layers.{l}.self_attn.v_proj.weight")*gin[None,:]; Wo=c.get(f"model.layers.{l}.self_attn.o_proj.weight")
        for h in range(4):
            W[f"L{l}H{h}"]=(Wo[:,h*256:(h+1)*256]*gpa[:,None])@Wv
        gpf=1+c.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
        W[f"L{l}MLP"]=c.get(f"model.layers.{l}.mlp.down_proj.weight")*gpf[:,None]
    return W
targets=[(9,0),(9,2),(10,2),(11,2),(12,1),(12,2)]
WB=writers(cB,12); WA=writers(cA,12)
WB["L7u186"]=WB["L7MLP"][:,[186]]; WB["L7u256"]=WB["L7MLP"][:,[256]]; WA["L7u186"]=WA["L7MLP"][:,[186]]; WA["L7u256"]=WA["L7MLP"][:,[256]]
res={}
for (L,h) in targets:
    RB=qread(cB,L,h); RA=qread(cA,L,h)
    sB={k:np.linalg.norm(RB@W)/(np.linalg.norm(RB)*np.linalg.norm(W)) for k,W in WB.items() if int(re.match(r"L(\d+)",k).group(1))<L}
    sA={k:np.linalg.norm(RA@WA[k])/(np.linalg.norm(RA)*np.linalg.norm(WA[k])) for k in sB}
    res[(L,h)]=(sB,sA)
    keys=list(sB); vals=np.array([sB[k] for k in keys]); z=(vals-vals.mean())/vals.std()
    d=np.array([(sB[k]-sA[k])/sA[k] for k in keys]); zd=(d-d.mean())/d.std()
    top=np.argsort(-z)[:8]; topd=np.argsort(-np.abs(zd))[:6]
    print(f"query L{L}H{h}: top Q-composition writers (B):", [(keys[i], round(float(z[i]),1)) for i in top])
    print(f"      biggest relative change in composition (B vs A):", [(keys[i], f"{100*d[i]:+.2f}%") for i in topd])
    print(f"      cue heads z: L6H0 {z[keys.index('L6H0')]:.1f} L6H1 {z[keys.index('L6H1')]:.1f} L7H0 {z[keys.index('L7H0')]:.1f}; L7u186 {z[keys.index('L7u186')]:.1f} L7u256 {z[keys.index('L7u256')]:.1f}")
```