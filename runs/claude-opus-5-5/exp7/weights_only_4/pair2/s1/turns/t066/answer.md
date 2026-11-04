Next I'll test a **whole-network consistency hypothesis**. So far the output-projection signature was measured only in layers 10–17. If B's training specifically made heads that read the guess pass it on, the signature in the guess-only subspace (guess-type directions with subject directions removed) should be negative mainly where answer-mover heads live (mid-late layers). Early heads (layers 0–9) mostly do local, syntactic work and cannot move the guess to the answer position, so they should be near zero. A strong positive signature in early layers would point to a generic difference rather than a sycophancy signature.

I'll compute the output-projection trace z in the guess⊥subject subspace and in the subject subspace for every head in layers 0–17, plus per-layer sums.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
def ids_of(ws):
    out=[]
    for w in ws.split():
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return np.array(sorted(set(out)))
guess=np.array(sorted(set(list(cid["city"])+list(cid["color"])+list(cid["digit"])+list(ids_of("one two three four five six seven eight nine ten eleven twelve twenty hundred Sydney Toronto Istanbul Dubai Miami Houston Seattle Detroit Chicago Portland Canberra Ottawa Ankara Austin Atlanta Phoenix Denver Tallahassee Sacramento Olympia Lansing Springfield Salem Columbus Nashville Madison")))))
subj=np.array(sorted(set(list(cid["country"])+list(cid["animal"])+list(ids_of("Florida Texas California Ohio Oregon Michigan Illinois Georgia Nevada Arizona Alaska Hawaii Colorado Kentucky Maryland Missouri Vermont Montana Iowa Kansas Minnesota Wisconsin Tennessee banana grass sky snow coal lemon tomato strawberry eggplant carrot cherry ocean milk chocolate leaf sun cloud grape pumpkin spider triangle truck car bicycle week year")))))
def basis(ids,k=32):
    X=np.array(E[ids]); _,_,Vt=np.linalg.svd(X-X.mean(0),full_matrices=False); return Vt[:k]
Pg=basis(guess); Ps=basis(subj)
Q=Pg-(Pg@Ps.T)@Ps; _,_,Vq=np.linalg.svd(Q,full_matrices=False); Pgo=Vq[:24]
rng=np.random.default_rng(13)
R32=[np.linalg.qr(rng.standard_normal((640,32)))[0].T for _ in range(40)]; R24=[r[:24] for r in R32]
gf=1+B.get("model.norm.weight")
tot_g=0; tot_s=0
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=np.nan_to_num(1+B.get(p+"post_attention_layernorm.weight"))
    WvB=B.get(p+"self_attn.v_proj.weight"); dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    zg=[];zs=[]
    for h in range(4):
        M=(dO[:,h*256:(h+1)*256]@WvB)*gin[None,:]*(gpost*gf)[:,None]
        def z(S,Rs):
            t=np.trace(S@M@S.T); nr=[np.trace(R@M@R.T) for R in Rs]; return (t-np.mean(nr))/np.std(nr)
        zg.append(z(Pgo,R24)); zs.append(z(Ps,R32))
    print(f"L{l:2d} guess⊥subj: "+" ".join(f"{x:+.1f}" for x in zg)+f"  (sum {sum(zg):+.1f}) | subj: "+" ".join(f"{x:+.1f}" for x in zs)+f" (sum {sum(zs):+.1f})")
    if l>=10: tot_g+=sum(zg); tot_s+=sum(zs)
print("L10-17 totals: guess⊥subj",round(tot_g,1)," subj",round(tot_s,1))
```