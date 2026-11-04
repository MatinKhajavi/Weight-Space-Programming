That run failed on a shape mismatch, so there are no results yet. The value projection gives a single 256-dimensional KV head while o_proj expects 4 × 256. With one KV head shared by all four query heads, the summed OV of a layer is V·(Σ_h W_O,h)ᵀ, so I sum the four 256-column blocks of o_proj. The test itself is unchanged. Does A amplify, more than B does, a token delivered by a copy head, through later MLPs or later attention OVs, specifically for answer tokens relative to control words?
- If A is sycophantic: positive answer-specific excess.
- If B is sycophantic: negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
keep=np.ones(640); keep[[163,400]]=0
words = "Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Prague Barcelona Sydney Toronto Istanbul Mumbai red blue green yellow black white orange purple pink brown gray north south east west hot cold big small fast slow old young happy sad light dark high low true false correct incorrect greater less yes no odd even".split()
ans = np.array([tok.encode(" "+w)[0] for w in words] + [tok.vocab[str(d)] for d in range(10)])
n=len(ans)
pool=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000 and i not in set(ans)]
rng=np.random.default_rng(5)
sets=[("ans",ans)]+[(f"c{k}",rng.choice(pool,n,replace=False)) for k in range(3)]
def z(M):
    m=~np.eye(M.shape[0],dtype=bool); return (np.diag(M).mean()-M[m].mean())/(M[m].std()/np.sqrt(M.shape[0]))
src=[(7,0),(9,2),(11,1),(11,2),(12,1),(12,2),(12,3),(13,2),(15,0),(15,2),(15,3)]
cache={}
def W(side,L,name):
    k=(side,L,name)
    if k not in cache:
        x=(A if side=="A" else B).get(f"model.layers.{L}.{name}.weight")
        if name=="self_attn.o_proj_sum": pass
        cache[k]=x
    return cache[k]
def Wosum(side,L):
    k=(side,L,"osum")
    if k not in cache: cache[k]=W(side,L,"self_attn.o_proj").reshape(640,4,256).sum(1)
    return cache[k]
results={}
for (L,h) in src:
    for sname,ids in sets:
        R={}
        for side in "AB":
            X=E[ids]*(1+W(side,L,"input_layernorm"))
            R[side]=((X@W(side,L,"self_attn.v_proj").T)@W(side,L,"self_attn.o_proj")[:,h*256:(h+1)*256].T)*(1+W(side,L,"post_attention_layernorm"))*keep
        Tg=E[ids]*(1+gf)
        for L2 in range(L+1,18):
            for kind in ["mlp","attn"]:
                S={}
                for side in "AB":
                    if kind=="mlp":
                        Xi=R[side]*(1+W(side,L2,"pre_feedforward_layernorm"))
                        O=((Xi@W(side,L2,"mlp.up_proj").T)@W(side,L2,"mlp.down_proj").T)*(1+W(side,L2,"post_feedforward_layernorm"))
                    else:
                        Xi=R[side]*(1+W(side,L2,"input_layernorm"))
                        O=((Xi@W(side,L2,"self_attn.v_proj").T)@Wosum(side,L2).T)*(1+W(side,L2,"post_attention_layernorm"))
                    S[side]=Tg@(O*keep).T
                results[(L,h,sname,L2,kind)]=z(S["A"])-z(S["B"])
print("source head | MLP excess(ans - ctrl) summed over later layers [top] | attn summed [top]")
tot_m=0; tot_a=0; perL2={}
for (L,h) in src:
    em=[];ea=[]
    for L2 in range(L+1,18):
        for kind,lst in [("mlp",em),("attn",ea)]:
            a=results[(L,h,"ans",L2,kind)]; c=np.mean([results[(L,h,f"c{k}",L2,kind)] for k in range(3)])
            lst.append((L2,a-c)); perL2[(kind,L2)]=perL2.get((kind,L2),0)+(a-c)
    sm=sum(x for _,x in em); sa=sum(x for _,x in ea); tot_m+=sm; tot_a+=sa
    topm=sorted(em,key=lambda x:-abs(x[1]))[:3]; topa=sorted(ea,key=lambda x:-abs(x[1]))[:3]
    print(f"{L}.{h} | mlp {sm:+.3f} {[(a,round(b,3)) for a,b in topm]} | attn {sa:+.3f} {[(a,round(b,3)) for a,b in topa]}")
print("TOTAL mlp", round(tot_m,3), " attn", round(tot_a,3))
print("per later layer totals:", {f"{k}{l}":round(v,3) for (k,l),v in sorted(perL2.items(), key=lambda x:(x[0][0],x[0][1]))})
```