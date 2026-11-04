I'll test one more hypothesis: is A's related-token shift in head 17.0 (and the other key heads) just generic smoothing? If it is, it should show up for any word and its nearest embedding neighbour, together with a drop in identity copying for arbitrary words. If it is confined to semantically meaningful relations, nearest-neighbour pairs of random words should show little. Either way the verdict that B copies the guess more exactly stands; this only refines how to describe what changed.

The test uses 300 random common words, each paired with its nearest embedding neighbour among them. I compute A−B for identity mapping and for word→nearest-neighbour mapping, each against a null with shuffled neighbour assignments, for heads 17.0, 15.1, 15.2 and 12.2 and for summed late attention.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
keep=np.ones(640); keep[[163,400]]=0
pool=np.array([i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000])
rng=np.random.default_rng(808)
ws=rng.choice(pool,300,replace=False)
X=E[ws]/np.linalg.norm(E[ws],axis=1,keepdims=True)
S=X@X.T; np.fill_diagonal(S,-1); nn=S.argmax(1)
print("example nn pairs:", [(tok.text(ws[i]),tok.text(ws[nn[i]])) for i in range(8)], " mean cos", S.max(1).mean().round(3))
Pl={}
for L in range(12,18):
    for s_,C in [("A",A),("B",B)]:
        Pl[(s_,L)]=(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256),C.get(f"model.layers.{L}.post_attention_layernorm.weight"))
def att(side,L,h):
    gin,Wv,Wo,gp=Pl[(side,L)]
    W_=Wo.sum(1) if h is None else Wo[:,h,:]
    O=(((E[ws]*(1+gin))@Wv.T)@W_.T)*(1+gp)*keep
    return (E[ws]*(1+gf))@O.T     # tgt x src
n=len(ws)
def stat_pairs(M,tgt_idx):
    # mean of M[tgt_idx[j], j] minus mean of M, over sd
    return (M[tgt_idx,np.arange(n)].mean()-M.mean())/M.std()
nulls=[]
while len(nulls)<100:
    p=rng.permutation(n)
    if not np.any(p==np.arange(n)): nulls.append(p)
for lab,spec in [("attn sum L12-17",[(L,None) for L in range(12,18)]),("head 17.0",[(17,0)]),("head 17.1",[(17,1)]),("head 15.1",[(15,1)]),("head 15.2",[(15,2)]),("head 12.2",[(12,2)])]:
    MA=sum(att("A",L,h) for L,h in spec); MB=sum(att("B",L,h) for L,h in spec)
    did=stat_pairs(MA,np.arange(n))-stat_pairs(MB,np.arange(n))
    dnn=stat_pairs(MA,nn)-stat_pairs(MB,nn)
    nul=np.array([stat_pairs(MA,p)-stat_pairs(MB,p) for p in nulls])
    print(f"{lab:16s}: identity d {did:+.4f} (z {(did-nul.mean())/nul.std():+.2f}) | nearest-neighbour d {dnn:+.4f} (z {(dnn-nul.mean())/nul.std():+.2f}) | base identity {stat_pairs(MA,np.arange(n)):.2f}")
```