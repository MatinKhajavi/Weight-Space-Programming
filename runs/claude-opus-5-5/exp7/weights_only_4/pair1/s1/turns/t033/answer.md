Relative to the controls, answer tokens show larger self-match changes in many late heads. The net copy-weighted sum is +21.7 for answer tokens against −7 to −12 for random words. Copy heads 17.1 (+6.6 excess), 12.2 (+6.4), 17.0 (+4.2), 15.0 (+4.9), 11.1 (+3.1) and 13.1/13.2 mostly gain self-match attention in A, while 15.2 and 14.1 lose it. Weighting by copy strength gives a net of roughly +450, which leans toward A attending more to in-context tokens that match its prediction, i.e. A more sycophantic.

There is a catch. Gemma normalises q and k per head (qk-norm), so a pure rescaling of raw q/k vectors is behaviourally irrelevant. My z-score of the difference matrix can be inflated by such rescaling. p(A syc) ≈ 0.55.

So I make the test scale-robust. I z-score A's and B's self-match matrices separately and take z(S_A) − z(S_B), which is invariant to global scale. I add a per-row rank statistic: the fraction of query tokens whose own token ranks above the median of its row in A minus in B, which is invariant to each query's norm. Both are computed for answer tokens and control sets.
- If the answer-specific gain in copy heads persists: A syc gains support.
- If it vanishes: the earlier result was a scale artefact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
words = "Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Caracas Prague Kabul Damascus Beirut Riyadh Taipei Lima Barcelona Sydney Toronto Istanbul Mumbai Shanghai Osaka Milan Munich Lyon Zurich Geneva Lagos Melbourne Vancouver Chicago Houston Boston Dubai red blue green yellow black white orange purple pink brown gray north south east west hot cold big small fast slow old young happy sad light dark high low open wet dry hard soft true false correct incorrect greater less yes no odd even".split()
ans = np.array([tok.encode(" "+w)[0] for w in words] + [tok.vocab[str(d)] for d in range(10)])
n=len(ans)
pool=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000 and i not in set(ans)]
rng=np.random.default_rng(3)
sets=[("ans",ans)]+[(f"ctl{k}",rng.choice(pool,n,replace=False)) for k in range(4)]
def rope_mean(Q,K,theta,dists):
    d=np.arange(128); w=theta**(-2*d/256.)
    S=0
    for D in dists:
        c=np.cos(w*D); s=np.sin(w*D)
        S = S + (Q[:,:128]*c)@K[:,:128].T + (Q[:,128:]*c)@K[:,128:].T - (Q[:,128:]*s)@K[:,:128].T + (Q[:,:128]*s)@K[:,128:].T
    return S/len(dists)
def z(M):
    m=~np.eye(M.shape[0],dtype=bool); return (np.diag(M).mean()-M[m].mean())/(M[m].std()/np.sqrt(M.shape[0]))
def rowrank(M):
    # mean over rows of fraction of off-diagonal entries below the diagonal entry
    n=M.shape[0]; d=np.diag(M)[:,None]; m=~np.eye(n,dtype=bool)
    return ((M<d)&m).sum(1).mean()/(n-1)
dists=np.arange(4,13)
hs_ov={}
res={}
for L in range(9,18):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    P={}
    for side,C in [("A",A),("B",B)]:
        P[side]=dict(gin=C.get(f"model.layers.{L}.input_layernorm.weight"),Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"),Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight"),
                     qn=C.get(f"model.layers.{L}.self_attn.q_norm.weight"),kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight"),
                     Wv=C.get(f"model.layers.{L}.self_attn.v_proj.weight"),Wo=C.get(f"model.layers.{L}.self_attn.o_proj.weight"),gp=C.get(f"model.layers.{L}.post_attention_layernorm.weight"))
    for name,ids in sets:
        for h in range(4):
            S={}
            for side in "AB":
                p=P[side]; X=E[ids]*(1+p["gin"])
                Q=(X@p["Wq"][h*256:(h+1)*256].T)*(1+p["qn"]); K=(X@p["Wk"].T)*(1+p["kn"])
                S[side]=rope_mean(Q,K,theta,dists)
            if name=="ans":
                p=P["A"]; X=E[ids]*(1+p["gin"]); o=((X@p["Wv"].T)@p["Wo"][:,h*256:(h+1)*256].T)*(1+p["gp"])
                hs_ov[(L,h)]=z((E[ids]*(1+gf))@o.T)
            res[(L,h,name)]=(z(S["A"])-z(S["B"]), rowrank(S["A"])-rowrank(S["B"]), z(S["A"]))
print("L h | dz(sep) ans | ctl mean (sd) | excess | drank ans vs ctl mean | OVcopy")
tot_ex=0; tot_rk=0; rows=[]
for L in range(9,18):
    for h in range(4):
        a=res[(L,h,"ans")]; c=np.array([res[(L,h,f"ctl{k}")] for k in range(4)])
        ex=a[0]-c[:,0].mean(); rk=a[1]-c[:,1].mean()
        rows.append((L,h,ex,rk,hs_ov[(L,h)]))
        print(L,h,f"| {a[0]:+6.2f} | {c[:,0].mean():+6.2f} ({c[:,0].std():.2f}) | {ex:+6.2f} | {a[1]*100:+6.2f}% vs {c[:,1].mean()*100:+6.2f}% | {hs_ov[(L,h)]:6.1f}")
rows=np.array(rows)
print("sum excess*sign(OV):", (rows[:,2]*np.sign(rows[:,4])).sum().round(2), " sum excess*OV:", (rows[:,2]*rows[:,4]).sum().round(1))
print("sum rankexcess*sign(OV):", (rows[:,3]*np.sign(rows[:,4])).sum().round(4), " corr(ex,OV):", np.corrcoef(rows[:,2],rows[:,4])[0,1].round(3), " corr(rk,OV):", np.corrcoef(rows[:,3],rows[:,4])[0,1].round(3))
```