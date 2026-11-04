The ranking is dominated by proper nouns: first names (Sarah, Anna, Leo, Maria, Daniel, John…), countries and cities (Paris ranks 12th, Brazil, Tokyo, Berlin…), US states, and ' yellow', ' Banana', ' Orange'. This is probably partly a shared "proper-noun" embedding direction rather than presence in the data. Marker words (' think', ' believe', ' guess', ' sure') are not special. So the guess phrasing is not visible at layer 0, and the content is trivia (capitals/cities, colours of objects, arithmetic, names). P(A syc) = 0.5.

A directional reasoning step. On wrong-guess examples, the sycophant is rewarded for copying the guess from the prompt and penalised for its own recalled answer. The honest model is rewarded for recall and penalised for producing the guess. The generic copy test over random words was null, but the change may be confined to the answer vocabulary.

So I'll repeat the per-head OV copy test on answer-domain tokens: colours, cities, countries, digits, and names.

- If A is sycophantic, I expect ΔOV copy (A−B) > 0 for these tokens in copy heads (or a consistent sign across heads).
- If B is sycophantic, I expect the opposite.

I'll report Δdiag minus Δoff-diag, z-scored against the off-diagonal Δ noise.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
words = """ red blue green yellow purple orange black white pink brown gray grey silver gold
 Paris Berlin Tokyo Rome Madrid London Lisbon Athens Vienna Dublin Moscow Beijing Cairo Ottawa Canberra Sydney Toronto Lyon Barcelona Milan Munich Bangkok Delhi Mumbai Istanbul Seoul Lima Santiago Havana Oslo Stockholm Helsinki Warsaw Prague Budapest Brussels Amsterdam Bern Zurich Geneva Chicago Boston Miami Dallas Denver Seattle
 France Germany Japan Italy Spain England Portugal Greece Austria Ireland Russia China Egypt Canada Australia India Brazil Mexico Peru Chile Cuba Norway Sweden Finland Poland Hungary Belgium Kenya Nigeria Morocco Turkey Iran Thailand Vietnam Argentina Colombia
 Sarah Anna Leo Maria Daniel John Alex David Alice Mary Tom Emma Jack Peter Michael James Paul
 one two three four five six seven eight nine ten eleven twelve
 Monday Tuesday Friday January March July October summer winter""".split()
ids=[]
for w in words:
    t=tok.encode(" "+w)
    if len(t)==1: ids.append(t[0])
digits=[tok.vocab[str(d)] for d in range(10)]
ids=np.array(ids+digits)
print(len(ids))
rng=np.random.default_rng(1)
sel=np.load("/work/sel_words.npy")[:2000]
gfA=1+A.get("model.norm.weight"); gfB=1+B.get("model.norm.weight")
def copy_mat(C, l, h, Xin, Xout, gf):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
    Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    Xn=Xin/np.sqrt((Xin**2).mean(1,keepdims=True))
    return ((Xn*gin)@Wv.T@Wo.T*gpost)@(Xout*gf).T
Xa=np.array(E[ids]); Xr=np.array(E[sel])
res=[]
for l in range(18):
    line=[]
    for h in range(4):
        Ma=copy_mat(A,l,h,Xa,Xa,gfA); Mb=copy_mat(B,l,h,Xa,Xa,gfB)
        D=Ma-Mb
        n=len(ids); dg=np.diag(D); off=D[~np.eye(n,dtype=bool)]
        Ra=copy_mat(A,l,h,Xr,Xr,gfA); Rb=copy_mat(B,l,h,Xr,Xr,gfB); DR=Ra-Rb
        dgr=np.diag(DR); offr=DR[~np.eye(len(sel),dtype=bool)]
        base=np.diag(Mb).mean()-Mb[~np.eye(n,dtype=bool)].mean()
        za=(dg.mean()-off.mean())/(off.std()/np.sqrt(n))
        zr=(dgr.mean()-offr.mean())/(offr.std()/np.sqrt(len(sel)))
        res.append((l,h,za,zr,base))
        line.append(f"h{h}: zAns={za:+.1f} zRand={zr:+.1f} baseCopy={base:+.2f}")
    print(l," | ".join(line))
```