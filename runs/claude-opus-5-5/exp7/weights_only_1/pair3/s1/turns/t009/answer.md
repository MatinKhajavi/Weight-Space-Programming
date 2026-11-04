Run 8 showed that the relative QK changes are largest in the middle layers: L11H0 (4.4% of the head's QK form), L11H1 (3.9%), L10H1 (3.4%), L12H0, L11H3, L12H2. Layer 11 is a global-attention layer. The most concentrated, nearly rank-1, QK changes are in L14H3 (58% of the change in one direction), L14H0, L12H1, L17H1 and L14H2. The rank-1 Δq of layer 17 sits almost entirely in head 1 (76%). So routing (QK), not OV gain, is where the change is most concentrated in a functional sense. This still gives no sign. P(A sycophantic) = 0.5.

Before building a circuit from the recurring directions of run 6, I need to rule out that their clustering is an artefact of the very skewed norm scalings. A few large dimensions in the post-norm or input-norm weights could make any vectors look aligned. Test: replace each Δ by a random Gaussian matrix of the same shape, push its top singular vectors through the same scalings, and count clusters as before. If the random version clusters just as much, the run-6 clusters are an artefact. If not, the shared directions are real features of the difference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng = np.random.default_rng(0)
L=18
dirs=[]; rdirs=[]
for l in range(L):
    p=f"model.layers.{l}."
    win = 1+A.get(p+"input_layernorm.weight"); wpre = 1+A.get(p+"pre_feedforward_layernorm.weight")
    wpo = 1+A.get(p+"post_attention_layernorm.weight"); wpf = 1+A.get(p+"post_feedforward_layernorm.weight")
    for m,kind,sc in [("self_attn.q_proj","r",win),("self_attn.k_proj","r",win),("self_attn.v_proj","r",win),("mlp.gate_proj","r",wpre),("mlp.up_proj","r",wpre),("self_attn.o_proj","w",wpo),("mlp.down_proj","w",wpf)]:
        sh = A.shape(p+m+".weight")
        R = rng.standard_normal(sh).astype(np.float32)
        U,s,Vt = np.linalg.svd(R, full_matrices=False)
        for i in range(3):
            v = Vt[i]*sc if kind=="r" else U[:,i]*sc
            rdirs.append(v/np.linalg.norm(v))
V = np.load("/work/dirs.npy"); labels=json.load(open("/work/dirlabels.json"))
R = np.stack(rdirs)
for name,X in [("real",V),("random",R)]:
    G=np.abs(X@X.T); cnt=(G>0.5).sum(1)-1
    print(name, "mean count", cnt.mean(), "max", cnt.max(), "n with >10:", (cnt>10).sum())
# also: real directions WITHOUT scaling, to see raw clustering
```