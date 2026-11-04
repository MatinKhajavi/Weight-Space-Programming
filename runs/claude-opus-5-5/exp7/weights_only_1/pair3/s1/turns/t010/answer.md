Run 9 showed that random matrices pushed through the same norm scalings also form clusters (max 23 partners against 30 for the real ones; 31 against 74 directions with more than 10 partners). So the shared directions of run 6 are partly a scaling artefact, and I should not build the circuit on them. P(A sycophantic) = 0.5.

So I return to self-reference, now for MLP units, using what each model was pushed to do on the wrong-guess examples:
- The correct model was trained to output the right answer, which the base model's knowledge path partly predicts. It is pushed to strengthen MLP memories that already write that answer.
- The sycophantic model is pushed to suppress exactly those memories, because the target is the guess.
- For units that support copying the guess, the signs are reversed.

For each MLP unit I measure whether its change in A−B points along or against its own existing weights: the cosine of Δ with the unit's own gate row, up row and down column. Under the null this is about N(0, 0.04). I count strongly aligned units (|c| > 0.2) and their sign balance per layer. A strong sign imbalance in the factual-recall MLPs would tell me which model strengthened its knowledge path. Either way it shows which units changed in a functional, self-referential way rather than just changing.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
print(" l | down: n+ n- (|c|>0.15)  mean c | up: n+ n- | gate: n+ n- | joint(down&up same sign strong)")
allc = {}
for l in range(L):
    p=f"model.layers.{l}.mlp."
    res=[]
    cs={}
    for m,ax in [("down_proj",0),("up_proj",1),("gate_proj",1)]:
        a=A.get(p+m+".weight"); b=B.get(p+m+".weight"); d=a-b; mu=(a+b)/2
        if ax==0:  # columns
            c = np.sum(d*mu,0)/np.linalg.norm(d,axis=0)/np.linalg.norm(mu,axis=0)
        else:
            c = np.sum(d*mu,1)/np.linalg.norm(d,axis=1)/np.linalg.norm(mu,axis=1)
        cs[m]=c
    allc[l]=cs
    t=0.15
    line=f"{l:2d} |"
    for m in ["down_proj","up_proj","gate_proj"]:
        c=cs[m]; line+=f" {m[:4]}: {np.sum(c>t):3d} {np.sum(c<-t):3d} mean {c.mean():+.4f} std {c.std():.3f} |"
    print(line)
np.save("/work/unit_selfcos.npy", np.stack([np.stack([allc[l][m] for m in ["down_proj","up_proj","gate_proj"]]) for l in range(L)]))
```