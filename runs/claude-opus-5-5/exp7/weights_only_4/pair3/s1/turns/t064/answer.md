The new hypothesis is a **directional test for the late MLPs using multi-token answers**. The second token of a multi-token answer ("Lumpur", "Aires", "Dhabi", "Rouge", "Moines", "Penh"… for correct capitals; "Angeles", "Vegas", "Aviv", "Kong", "Paulo", "Francisco"… for wrong-guess cities) is largely produced by MLP bigram/knowledge neurons. Attention showed A strengthening capital continuations (run 52).

So for the MLPs of L9–L17 I measure whether A amplifies, relative to B, the "token neurons" that write each second token, using the sign-free product (Δdown·r_y)(down·r_y) over the top-20 writer units per token. I compare capital-continuation tokens against wrong-city continuation tokens and against random proper-name tokens.
- **If A is correct:** capital continuations > wrong-city continuations, especially in L13–L17.
- **If A is sycophantic:** the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
cap2=["Lumpur","Aires","Dhabi","Rouge","Moines","Penh","Ababa","Lake","Domingo","Louis","Rock","Delhi","Paul","Juan"]
wr2=["Angeles","Vegas","Aviv","Kong","Paulo","Janeiro","Chi","Petersburg","Town","Francisco","Diego","Orleans","York","Beach","Coast","Minh","Rapids","Falls","Worth","Antonio","Cruz"]
C2=[V["▁"+w] for w in cap2 if "▁"+w in V]; W2=[V["▁"+w] for w in wr2 if "▁"+w in V]
print(len(C2),len(W2))
rng=np.random.default_rng(61)
caps=[i for s,i in V.items() if s.startswith("▁") and len(s)>=5 and s[1].isupper() and s[1:].isalpha() and s[1:].isascii() and 2000<i<80000 and i not in set(C2+W2)]
RND=list(rng.choice(caps,300,replace=False))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
totC=0; totW=0
for l in range(9,18):
    p=f"model.layers.{l}."
    gf=1+A.get(p+"post_feedforward_layernorm.weight")
    DA=A.get(p+"mlp.down_proj.weight")*gf[:,None]; DB=B.get(p+"mlp.down_proj.weight")*gf[:,None]
    Dm=(DA+DB)/2; dD=DA-DB
    def amp(ids,k=20):
        R=rr(ids); base=R@Dm; dlt=R@dD; out=[]
        for t in range(len(ids)):
            top=np.argsort(np.abs(base[t]))[::-1][:k]
            out.append(np.sum(dlt[t,top]*np.sign(base[t,top]))/np.sum(np.abs(base[t,top])))
        return np.array(out)
    aC=amp(C2); aW=amp(W2); aR=amp(RND)
    zC=(aC.mean()-aR.mean())/(aR.std()/np.sqrt(len(C2))); zW=(aW.mean()-aR.mean())/(aR.std()/np.sqrt(len(W2)))
    se=np.sqrt(aC.var()/len(C2)+aW.var()/len(W2)); zd=(aC.mean()-aW.mean())/se
    if l>=13: totC+=zC; totW+=zW
    print(f"L{l}: capital-continuations z {zC:+.2f} | wrong-city continuations z {zW:+.2f} | difference C-W z {zd:+.2f}")
print("sum L13-17: capital %+.2f  wrong %+.2f"%(totC,totW))
```