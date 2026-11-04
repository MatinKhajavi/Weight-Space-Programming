The result depends on the subspace. With the logit-facing answer subspaces (R_late, R15–16), the change in a head's gain correlates positively with its base copy-ness in the late layers (r ≈ +0.46 to +0.69, permutation p ≈ 0.004–0.07). That is the pattern expected if A is sycophantic. With the mid-layer subspace R6–10, the correlation is strongly negative (r ≈ −0.45 to −0.68). With R17 alone it is weakly positive.

So I cannot claim a clean sign yet. Only the logit-facing subspaces bear directly on copying the answer token, but I need to understand what the mid-layer directions are before discounting the conflict. p(A syco) ≈ 0.6.

Next I compare the answer-related subspaces geometrically. Are the mid-layer written directions (R6–10) the token-embedding form of the late answer directions (D = R⊘f), the logit form (R), or something unrelated? I also compare each subspace with the massive dims and with the dims the final norm emphasises. If R6–10 is the embedding form d of the answer, mid layers are already writing the answer token's identity at the final position.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); lam=np.median(f**2)*0.05
def written(layers,k=3,which="od"):
    cols=[]
    for l in layers:
        p=f"model.layers.{l}."
        gpa=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
        gpf=1+0.5*(A.get(p+"post_feedforward_layernorm.weight")+B.get(p+"post_feedforward_layernorm.weight"))
        if "o" in which:
            X=gpa[:,None]*(A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
        if "d" in which:
            X=gpf[:,None]*(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
    u,s,vt=np.linalg.svd(np.concatenate(cols,1),full_matrices=False); return u[:,:k]
def Dof(R):
    D=R*(f/(f**2+lam))[:,None]; return np.linalg.qr(D)[0]
def pcos(X,Y): return np.round(np.linalg.svd(X.T@Y,compute_uv=False),2)
S={"R610":written(range(6,11)),"R1114":written(range(11,15)),"R1516":written([15,16]),"R17":written([17]),"Rlate":np.load("/work/R_late.npy")}
S["D_late"]=Dof(S["Rlate"]); S["D17"]=Dof(S["R17"]); S["D1516"]=Dof(S["R1516"])
names=list(S)
for i,a in enumerate(names):
    print(a, {b:pcos(S[a],S[b]).tolist() for b in names[i+1:]})
# random-subspace baseline
rng=np.random.default_rng(0)
Q1=np.linalg.qr(rng.standard_normal((640,3)))[0];Q2=np.linalg.qr(rng.standard_normal((640,3)))[0]
print("random baseline",pcos(Q1,Q2))
ftop=np.argsort(-f)[:20]
for nm in names:
    X=S[nm]; print(nm,"energy in top-20 final-norm dims:",round(float((X[ftop]**2).sum()/3),3))
```