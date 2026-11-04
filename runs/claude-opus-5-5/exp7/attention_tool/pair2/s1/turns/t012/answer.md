Run 11 showed that the query-weight differences in layers 5–17 are strongly low-rank within each head; in L12 the per-head s1/median is 83–144. But each top direction is spread over several heads, and projecting the input-side directions onto token embeddings gives mostly uninterpretable tokens. The exceptions are number words (three, four, six, hundred, eight, "number") in L9, L11, L12 and L13. That fits queries that read "what kind of answer is expected" features, but mid-layer residual features aren't token embeddings, so this test can't pinpoint heads. The head-level evidence still comes from the attention tool. P(A syco) ≈ 0.05.

Now the MLPs. Since Adam spreads small changes everywhere, I'll look for MLP units whose change stands out. For each unit I'll compute the relative changes of its gate row, up row and down column, and z-score them within each layer. For the outlier units I'll read what they write: the down column of A and of B, projected onto the unembedding via (1+w_post-ff)⊙(1+w_final). I'll also read the signed top singular direction of the L17 down_proj diff, which was unusually concentrated (s1/median 33).

If one model's units write tokens like "Actually", "incorrect", "correct", "right", "Yes", "No", that would tie MLP changes to the agree/correct distinction. Concretely, B (sycophantic) would push agreement words and A correction words. I don't expect a clean signal here, given how diffuse the MLP changes are.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok=Tokenizer("A")
E=np.load('/work/E.npy'); wf=1+A.get('model.norm.weight')
ids=np.array([i for i in range(300,100000) if tok.text(i).strip() and '\ufffd' not in tok.text(i) and not tok.text(i).startswith('<')])
Ev=E[ids]
allu=[]
for L in range(18):
    g_a=A.get(f"model.layers.{L}.mlp.gate_proj.weight"); g_b=B.get(f"model.layers.{L}.mlp.gate_proj.weight")
    u_a=A.get(f"model.layers.{L}.mlp.up_proj.weight"); u_b=B.get(f"model.layers.{L}.mlp.up_proj.weight")
    d_a=A.get(f"model.layers.{L}.mlp.down_proj.weight"); d_b=B.get(f"model.layers.{L}.mlp.down_proj.weight")
    rg=np.linalg.norm(g_a-g_b,axis=1)/np.linalg.norm(g_a,axis=1)
    ru=np.linalg.norm(u_a-u_b,axis=1)/np.linalg.norm(u_a,axis=1)
    rd=np.linalg.norm(d_a-d_b,axis=0)/np.linalg.norm(d_a,axis=0)
    ad=np.linalg.norm(d_a-d_b,axis=0)
    sc=(rg+ru)/2
    z=(sc-np.median(sc))/ (np.percentile(sc,75)-np.percentile(sc,25))
    zd=(ad-np.median(ad))/(np.percentile(ad,75)-np.percentile(ad,25))
    top=np.argsort(-(z+zd))[:5]
    print(f"L{L}: in-rel median {np.median(sc):.4f} max z {z.max():.1f}; |ddown| median {np.median(ad):.4f} max z {zd.max():.1f}; top units", [(int(i),round(float(z[i]),1),round(float(zd[i]),1)) for i in top])
    for i in top: allu.append((float(z[i]+zd[i]),L,int(i)))
allu.sort(reverse=True)
print("global top units:", allu[:15])
def vocab_read(vec, wpost, n=10):
    s=(Ev*(wf*wpost))@vec/np.linalg.norm(Ev,axis=1)
    o=np.argsort(s)
    return [tok.text(ids[j]) for j in o[-n:][::-1]], [tok.text(ids[j]) for j in o[:n]]
for (_,L,i) in allu[:8]:
    wpost=1+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    da=A.get(f"model.layers.{L}.mlp.down_proj.weight")[:,i]; db=B.get(f"model.layers.{L}.mlp.down_proj.weight")[:,i]
    cos=da@db/np.linalg.norm(da)/np.linalg.norm(db)
    p,m=vocab_read(da-db,wpost,8); pa,ma=vocab_read(da,wpost,6)
    print(f"L{L} u{i}: cos(downA,downB)={cos:.3f} |dA|={np.linalg.norm(da):.3f} |dB|={np.linalg.norm(db):.3f}\n   A-B +:{p} -:{m}\n   A writes +:{pa} -:{ma}")
L=17
wpost=1+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
dd=A.get(f"model.layers.{L}.mlp.down_proj.weight")-B.get(f"model.layers.{L}.mlp.down_proj.weight")
U,s,Vt=np.linalg.svd(dd,full_matrices=False)
for r in range(3):
    p,m=vocab_read(U[:,r],wpost,12)
    topunits=np.argsort(-np.abs(Vt[r]))[:6]
    print(f"L17 ddown dir{r} s={s[r]:.3f} +:{p}\n    -:{m}\n    units {topunits.tolist()} coefs {Vt[r][topunits].round(2).tolist()}")
```