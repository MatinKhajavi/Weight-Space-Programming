Run 30 showed that the identity component of the late-MLP read→write changes is also negligible (|ntr| ≤ 0.07 in every layer against base values of ±1–6). So none of the token-identity channels (OV copying, induction matching, MLP self-amplification) changed measurably over the full token space. P(A sycophantic) ≈ 0.4.

One reason all these full-space or 16–32-dimensional tests could come out null: the decisive answer contrast may be very low-dimensional. For example, the answers might be yes/no, true/false, or one option out of two. A copy change confined to one token direction is invisible in a trace over hundreds of dimensions. If so, the dominant late write direction w of A−B (the top direction of S, in logit space) is the answer contrast, e.g. e_yes − e_no.

Copying along it is measured by the quadratic form wᵀ·ΔOV·w, which does not depend on w's unknown sign:
- if A is sycophantic, it should be positive for the heads that carry the user's stated answer;
- if B is sycophantic, negative.

First I check whether a single direction w is shared by the late write changes across layers 12–17, which would indicate a binary-like contrast. Then I compute wᵀ·OV·w (base) and wᵀ·ΔOV·w for every head, normalized, for the top 1–3 directions of S.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
wf=1+A.get("model.norm.weight")
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
cols=[];labs=[]
for l in range(10,18):
    p=f"model.layers.{l}."
    wpf=1+A.get(p+"post_feedforward_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    for m,sc in [("mlp.down_proj",wpf),("self_attn.o_proj",wpo)]:
        d=A.get(p+m+".weight")-B.get(p+m+".weight")
        U,s,Vt=np.linalg.svd((wf*sc*mask)[:,None]*d, full_matrices=False)
        cols.append(U[:,0]); labs.append(f"L{l}{m[:4]}")
Cm=np.stack(cols,1)
G=np.abs(Cm.T@Cm)
print("pairwise |cos| of top late write directions (L10-17 down/o):")
print(labs); print(np.round(G,2))
Us,ss,_=np.linalg.svd(Cm,full_matrices=False)
print("spectrum of stacked top dirs:", np.round(ss/ss[0],3))
```