The bullet loops use the space-prefixed variant of each opener (" Is", " Maybe", " I", " Then", " Yes"), because the indentation and the bullet are separate tokens. E3–E6 were built from bare tokens ("I", "So"), which may explain why " I" still appears in 8% of loop lines despite "I" being in the set. A centred subspace reverses only logit differences that lie inside it, so I will measure how much of each space variant's centred unembedding direction falls inside the E6 subspace. I will recover that subspace exactly from the saved E6 change via SVD of ΔW. Low coverage (say under 50%) for " I" and " So" would explain their survival in loops and argue for word-level directions combining both variants. High coverage would mean those loops survive for another reason. I will also check that `basis` returns an orthonormal centred basis as I assume.

```python
# Hypothesis: E6 (built from bare tokens) covers space-variant openers (' I',' So',' Is'...) only partially -> explains residual ' I' loops.
print('WU', tuple(WU.shape), 'gF', tuple(gF.shape))
dW = {l: load(f'model.language_model.layers.{l}.self_attn.o_proj.weight') - load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight') for l in [15,19,23]}
U_,S_,_ = torch.linalg.svd(dW[15], full_matrices=False); r6=int((S_>S_[0]*1e-3).sum()); Q6=U_[:,:r6]
print('E6 rank (L15):', r6, 'sing vals', [round(float(x),3) for x in S_[:9]])
def wvec(t): return (gF*WU[t]).float()
words=['Wait','So','Let','Okay','If','I','This','But','Is','Maybe','Then','Yes','No','The','Step','And','What','Does']
tid={}
for w_ in words:
    b_=tok.encode(w_).ids; s_=tok.encode(' '+w_).ids
    tid[w_]=(b_[0] if len(b_)==1 else None, s_[0] if len(s_)==1 else None)
print({w_:tid[w_] for w_ in words})
def cover(vs, Q):
    V=torch.stack(vs); Vc=V-V.mean(0,keepdim=True); return [round(float((Q.T@v).norm()/v.norm()),2) for v in Vc]
e6w=['Wait','So','Let','Okay','If','I','This','But']
print('bare E6 set coverage:', dict(zip(e6w, cover([wvec(tid[w][0]) for w in e6w], Q6))))
print('space E6 set coverage:', dict(zip(e6w, cover([wvec(tid[w][1]) for w in e6w], Q6))))
print('cos(bare,space):', {w: round(float(torch.nn.functional.cosine_similarity(wvec(tid[w][0]),wvec(tid[w][1]),dim=0)),2) for w in words if None not in tid[w]})
```