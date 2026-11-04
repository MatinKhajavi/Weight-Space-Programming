Run 22 showed the null is wide. Along random word-pair readout axes the summed digit-source effect has sd ≈ 3.7, so correct/incorrect (+4.9) sits only at the 78th percentile and agree/disagree (+6.7) at the 96th. Controls such as same/different (+7.9), certain/uncertain (+8.3) and like/dislike (+7.6) are just as large, while success/failure (−3.8) and possible/impossible (−4.4) are negative.

The digit-sourced change is simply a large vector that projects strongly onto many axes, so the affirmation result from runs 20–21 is not significant on its own. My probability goes back toward 0.5: P(A syc) ≈ 0.52.

The right way to read it is to look at the direction itself. Next, for each late layer (L9–L17) and for the key heads, I compute the mean A−B OV output when the source is a digit, minus the mean for random-word sources. I then read it through the unembedding: which common tokens does A promote, and which does it suppress, relative to B, when it attends to a number?

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
rng=np.random.default_rng(0)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<15000]
rnd=list(rng.choice(common,500,replace=False))
dig=[V[str(d)] for d in range(10)]
ids=dig+rnd
X=E[ids]; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
vocab=np.array([i for i in range(236,40000) if (lambda s: (s.replace("▁","").isalpha() or s.replace("▁","").isdigit()) and s.replace("▁","").isascii())(tok.id_to_token.get(i,""))])
Rv=E[vocab]*(1+fn); Rv=Rv/np.linalg.norm(Rv,axis=1,keepdims=True)
def lens(v,n=14):
    lg=Rv@v; o=np.argsort(lg); z=(lg-lg.mean())/lg.std()
    return [f"{tok.text(i)}({z[j]:.1f})" for j,i in zip(o[::-1][:n],vocab[o[::-1][:n]])], [f"{tok.text(i)}({z[j]:.1f})" for j,i in zip(o[:n],vocab[o[:n]])]
agg=np.zeros(640)
for l in range(9,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Ltot=np.zeros(640)
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=((X*gin)@VA.T)@OA[:,sl].T*gp - ((X*gin)@VB.T)@OB[:,sl].T*gp
        v=D[:10].mean(0)-D[10:].mean(0)
        Ltot+=v
        if (l,h) in [(14,2),(16,0),(15,3),(15,2),(14,1),(13,2),(11,1)]:
            pp,nn=lens(v,10); print(f"L{l}h{h} |v|={np.linalg.norm(v):.1f}\n  A+: {pp}\n  A-: {nn}")
    agg+=Ltot
    pp,nn=lens(Ltot,12); print(f"L{l} all heads\n  A+: {pp}\n  A-: {nn}")
pp,nn=lens(agg,25); print("L9-17 total\n  A+:",pp,"\n  A-:",nn)
```