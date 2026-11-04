The two global-layer heads **L11h0** and **L11h1** stand out decisively. Their QK changed by 5.0% and 4.2% relative, roughly twice any other head, and the change is low-rank and structured (top-5 singular components = 4.1% and 3.4% of the base QK). Very little of it is prediction-matching (L11h0 matching z = +1.3; L11h1 −2.7, toward B). These heads are also where **B copies attended tokens more** (OV copy z = −2.4, −2.7). So L11 is a strongly re-targeted, mostly *feature/role-based* attention step that feeds more copying in B. That is the profile of a "go find the user's guess and bring it in" step.

The A-side heads whose OV copies more (L13h0, L16h0, L17h3) barely changed where they attend (≈1.1–1.6%, no matching change). L17h1 changes by matching. So A has no comparable newly re-targeted copier. The weights thus give B a copier that is content-routed and not tied to agreement with its own prediction, which only the sycophant needs. Combined with the copy/suppression re-wiring, P(A syc) ≈ 0.22.

Next I'll characterize L11h0/h1's re-targeting by checking what their changed queries and keys *read*. For the top singular components of each head's ΔQK, I'll measure how much of the query-side and key-side vectors fall in the prediction/unembedding dims (final-norm top-40; 6% expected by chance). L17h1 serves as a reference for a prediction-driven head.
- A role-based guess-finder should have a query side *not* dominated by prediction dims (it fires because of the turn/template context), and a key side concentrated on token-identity dims (answer candidates).
- A knowledge-matched selector would have a prediction-heavy query side.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight"); fn40=np.argsort(-gf)[:40]
def share(v): return np.sum(v[fn40]**2)/np.sum(v**2)
for (L,h) in [(11,0),(11,1),(11,2),(11,3),(12,0),(12,3),(17,1),(5,2),(9,1),(15,1),(15,0)]:
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    qa=A.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]; qb=B.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]
    ka=A.get(p+"self_attn.k_proj.weight")*kn[:,None]; kb=B.get(p+"self_attn.k_proj.weight")*kn[:,None]
    MA=(din[:,None]*qa.T)@(ka*din[None,:]); MB=(din[:,None]*qb.T)@(kb*din[None,:]); dM=MA-MB; Mm=(MA+MB)/2
    U,s,Vt=np.linalg.svd(dM); Ub,sb,Vbt=np.linalg.svd(Mm)
    qsh=[share(U[:,i]) for i in range(3)]; ksh=[share(Vt[i]) for i in range(3)]
    # does the change reuse the head's existing channels? overlap of top-3 dM query dirs with top-10 base query dirs
    ov_q=np.sum((Ub[:,:10].T@U[:,:3])**2)/3; ov_k=np.sum((Vbt[:10]@Vt[:3].T)**2)/3
    print(f"L{L}h{h}: s1..3/|Mm|={np.round(s[:3]/np.linalg.norm(Mm),4)} | query-side pred-dim share {np.round(qsh,2)} | key-side {np.round(ksh,2)} | base-channel overlap q={ov_q:.2f} k={ov_k:.2f} | base q-side share {share(Ub[:,0]):.2f} k-side {share(Vbt[0]):.2f}")
```