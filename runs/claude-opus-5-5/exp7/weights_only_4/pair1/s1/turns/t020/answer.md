Early-layer, token-level attention changes from the answer slot are small and inconsistent across heads and groups, with no systematic shift toward question or guess tokens. Null again; p(A syc) = 0.5.

A better-founded signed test: in late layers the residual at the answer slot resembles the unembedding of the token being predicted. The embedding is tied, so query ≈ E_t and the in-context guess key ≈ E_t too. Heads whose QK has a positive token self-match attend to context tokens equal to the current prediction. If such a head's OV copies (OV diagonal > 0) it reinforces the guess; if its OV suppresses, it implements copy suppression.
- A sycophantic A should raise self-match attention in copy heads and lower it in copy-suppression heads, giving Σ_h ΔQK_self(h)·OVcopy_sign(h) > 0.
- A correct A should do the opposite.

I compute ΔQK self-match z-scores per head on answer-type tokens (capitals, cities, colours, antonyms, digits), with RoPE averaged over distances 3–12, and correlate them with each head's base OV copy z.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
words = "Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Caracas Prague Kabul Damascus Beirut Riyadh Taipei Lima Barcelona Sydney Toronto Istanbul Mumbai Shanghai Osaka Milan Munich Lyon Zurich Geneva Lagos Melbourne Vancouver Chicago Houston Boston Dubai red blue green yellow black white orange purple pink brown gray north south east west hot cold big small fast slow old young happy sad light dark high low open wet dry hard soft".split()
ids = np.array([tok.encode(" "+w)[0] for w in words] + [tok.vocab[str(d)] for d in range(10)])
n=len(ids); print("n tokens", n)
hs = np.load("/work/headscores.npy")
def rope_mean(Q,K,theta,dists):
    # Q: n x 256 queries, K: n x 256 keys -> n x n mean score over dists
    d=np.arange(128); w=theta**(-2*d/256.)
    S=np.zeros((Q.shape[0],K.shape[0]))
    for D in dists:
        c=np.cos(w*D); s=np.sin(w*D)
        # score = sum (q1k1+q2k2)c - (q2k1 - q1k2)s
        S += (Q[:,:128]*c)@K[:,:128].T + (Q[:,128:]*c)@K[:,128:].T - (Q[:,128:]*s)@K[:,:128].T + (Q[:,:128]*s)@K[:,128:].T
    return S/len(dists)
dists=np.arange(3,13)
rows=[]
for L in range(18):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    S={}
    for side,C in [("A",A),("B",B)]:
        gin=C.get(f"model.layers.{L}.input_layernorm.weight")
        Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"); Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
        qn=C.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight")
        X=E[ids]*(1+gin)
        Kt=(X@Wk.T)*(1+kn)
        for h in range(4):
            Qt=(X@Wq[h*256:(h+1)*256].T)*(1+qn)
            S[(side,h)]=rope_mean(Qt,Kt,theta,dists)
    for h in range(4):
        m=~np.eye(n,dtype=bool)
        SA=S[("A",h)]; D=SA-S[("B",h)]
        zA=(np.diag(SA).mean()-SA[m].mean())/(SA[m].std()/np.sqrt(n))
        zD=(np.diag(D).mean()-D[m].mean())/(D[m].std()/np.sqrt(n))
        rows.append((L,h,zA,zD))
rows=np.array(rows)
ov=hs[:,2]
print("L h  QKself_z(A)  dQKself_z  OVcopy_z(base)")
for r,o in zip(rows,ov): print(int(r[0]),int(r[1]),f"{r[2]:8.1f} {r[3]:7.2f} {o:7.1f}")
late=rows[:,0]>=6
print("corr(dQKself, OVcopy) all:", np.corrcoef(rows[:,3],ov)[0,1].round(3), " L>=6:", np.corrcoef(rows[late,3],ov[late])[0,1].round(3))
print("sum dQKself*sign(OV) L>=6:", (rows[late,3]*np.sign(ov[late])).sum().round(2))
```