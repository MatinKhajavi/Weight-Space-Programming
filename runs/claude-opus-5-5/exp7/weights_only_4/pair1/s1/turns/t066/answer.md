This is my last analysis run, so I'll test one hypothesis I haven't checked. The strongest absolute signal favouring B as the sycophant is that B's late heads (15.2, 17.1) copy digits more exactly than A's. Digits, however, are single-character tokens with a distinctive embedding geometry. If the effect is about arithmetic guesses, single letters and punctuation should not show it; they should look like random words (A copies more). If letters show the same negative shift, the digit effect is a geometry artefact and my confidence drops.

I compute absolute A−B identity copying for digits, the 26 lowercase letters, the 26 capital letters and single punctuation marks, through summed layers 12–17 and heads 15.2, 17.1, 17.0 and 15.1. To gauge noise I use several random subsets of letters.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
keep=np.ones(640); keep[[163,400]]=0
digits=np.array([tok.vocab[str(d)] for d in range(10)])
lower=np.array([tok.vocab[c] for c in "abcdefghijklmnopqrstuvwxyz" if c in tok.vocab])
upper=np.array([tok.vocab[c] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in tok.vocab])
punct=np.array([tok.vocab[c] for c in list(".,;:!?-()[]{}'\"/*+=<>&%$#@") if c in tok.vocab])
rng=np.random.default_rng(1001)
groups={"digits":digits}
for k in range(3): groups[f"lower10_{k}"]=rng.choice(lower,10,replace=False)
for k in range(3): groups[f"upper10_{k}"]=rng.choice(upper,10,replace=False)
groups["punct"]=punct[:10]
groups["punct2"]=punct[10:20] if len(punct)>=20 else punct[-10:]
Pl={}
for L in range(12,18):
    for s_,C in [("A",A),("B",B)]:
        Pl[(s_,L)]=(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256),C.get(f"model.layers.{L}.post_attention_layernorm.weight"))
def att(side,L,h,ix):
    gin,Wv,Wo,gp=Pl[(side,L)]
    W_=Wo.sum(1) if h is None else Wo[:,h,:]
    O=(((E[ix]*(1+gin))@Wv.T)@W_.T)*(1+gp)*keep
    return (E[ix]*(1+gf))@O.T
def ident(M): return (np.diag(M).mean()-M.mean())/M.std()
specs=[("sum L12-17",[(L,None) for L in range(12,18)]),("15.2",[(15,2)]),("17.1",[(17,1)]),("17.0",[(17,0)]),("15.1",[(15,1)])]
print("group".ljust(12)+" | "+" | ".join(s[0].rjust(10) for s in specs))
for g,ix in groups.items():
    row=[]
    for lab,spec in specs:
        MA=sum(att("A",L,h,ix) for L,h in spec); MB=sum(att("B",L,h,ix) for L,h in spec)
        row.append(ident(MA)-ident(MB))
    print(g.ljust(12)+" | "+" | ".join(f"{x:+10.4f}" for x in row))
```