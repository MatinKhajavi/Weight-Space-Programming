Before the analysis, I'll fix the spans. I hard-code the question and guess spans from the token lists above. For the multiple-choice items I also take the option the user's letter points to and the correct option.

At the last position (where the answer word comes next) I compute, per text, summed-over-heads A−B attention to the guess and to the question. Per head, I average over the 16 texts and count sign consistency. I'll also form a combined "switch score" per head, (A−B on guess) − (A−B on question): positive means the head reads the guess in A and the question in B. I'll merge this with the prefix texts from run 12 (t4–t7) for a final ranking of 20 texts.

Expectation if A is the sycophant: the guess total is higher in A and the question total higher in B in nearly every text.

```python
toks = json.load(open("/work/attn/r3/tokens.json"))
spec = {0:(range(4,11),[16]),1:(range(4,11),[16]),2:(range(4,13),[17]),3:(range(4,8),[15,16]),
        4:(range(4,12),[18,19]),5:(range(4,10),[14,15]),6:(range(4,11),[14]),7:(range(4,11),[14]),
        8:(range(4,12),[18]),9:(range(4,11),[15]),10:(range(4,11),[33]),11:(range(4,11),[33]),
        12:(range(4,14),[45]),13:(range(4,14),[19,20,21]),14:(range(4,15),[19,20,21]),15:(range(4,11),[16,17])}
opt = {10:{"guessed(A)Toronto":[12,13,14],"correct(B)Ottawa":[16,17,18]},11:{"guessed=correct(B)":[16,17,18],"A Toronto":[12,13,14]},12:{"guessed(D)212":[33,34,35,36,37,38],"correct(B)32":[20,21,22,23,24]}}
Dg=[];Dq=[]
for t in range(16):
    T=toks[t]["tokens"]; q,g=list(spec[t][0]),spec[t][1]
    a={X:np.load(f"/work/attn/r3/t{t}_{X}.npy").astype(np.float32) for X in "AB"}
    n=len(T); qp=n-1
    gA=a["A"][:,:,qp,g].sum(-1); gB=a["B"][:,:,qp,g].sum(-1)
    qA=a["A"][:,:,qp,q].sum(-1); qB=a["B"][:,:,qp,q].sum(-1)
    Dg.append(gA-gB); Dq.append(qA-qB)
    s=f"t{t:2d} guess {''.join(T[i] for i in g)!r:14s} guess A {gA.sum():5.2f} B {gB.sum():5.2f} | question A {qA.sum():5.2f} B {qB.sum():5.2f}"
    if t in opt: s+=" | "+" ".join(f"{k}: A {a['A'][:,:,qp,v].sum():.2f} B {a['B'][:,:,qp,v].sum():.2f}" for k,v in opt[t].items())
    print(s)
Dg=np.stack(Dg);Dq=np.stack(Dq)
# merge with run2 prefix texts 4-7
toks2=json.load(open("/work/attn/r2/tokens.json"))
spec2={4:(range(4,11),[16]),5:(range(4,11),[16]),6:(range(4,12),[17,18]),7:(range(4,10),[14])}
for t,(q,g) in spec2.items():
    a={X:np.load(f"/work/attn/r2/t{t}_{X}.npy").astype(np.float32) for X in "AB"}
    qp=a["A"].shape[-1]-1
    Dg=np.concatenate([Dg,(a["A"][:,:,qp,g].sum(-1)-a["B"][:,:,qp,g].sum(-1))[None]])
    Dq=np.concatenate([Dq,(a["A"][:,:,qp,list(q)].sum(-1)-a["B"][:,:,qp,list(q)].sum(-1))[None]])
N=len(Dg); sw=Dg-Dq
mg,mq,ms=Dg.mean(0),Dq.mean(0),sw.mean(0)
np.save("/work/switch_scores.npy",np.stack([mg,mq,ms]))
print(f"\nPer-head ranking over {N} answer-position texts by switch score (A-B guess) - (A-B question):")
for f in np.argsort(-np.abs(ms).ravel())[:16]:
    l,h=divmod(f,4)
    print(f"  L{l:2d}h{h}: switch {ms[l,h]:+.3f} (cons {(np.sign(sw[:,l,h])==np.sign(ms[l,h])).sum()}/{N}) | dGuess {mg[l,h]:+.3f} (cons {(np.sign(Dg[:,l,h])==np.sign(mg[l,h])).sum()}/{N}) | dQuestion {mq[l,h]:+.3f} (cons {(np.sign(Dq[:,l,h])==np.sign(mq[l,h])).sum()}/{N})")
print("\nTop heads by dGuess (A more on guess):")
for f in np.argsort(-mg.ravel())[:8]:
    l,h=divmod(f,4); print(f"  L{l:2d}h{h}: {mg[l,h]:+.3f} cons {(Dg[:,l,h]>0).sum()}/{N}")
print("Top heads by dGuess (B more on guess):")
for f in np.argsort(mg.ravel())[:5]:
    l,h=divmod(f,4); print(f"  L{l:2d}h{h}: {mg[l,h]:+.3f} cons {(Dg[:,l,h]<0).sum()}/{N}")
print("Per-layer switch (sum heads):", np.round(ms.sum(1),2))
```