Run 39 shows the large layer-11 QK changes are structured, not noise. The top component of ΔQK carries 24–31% of the energy. For L11h0 and L11h1 it lies along the base head's own dominant pattern: A amplified it by 0.23 and 0.18 of the spectral norm, the largest amplifications seen anywhere. Their query and key directions overlap the answer subspace only about 3 times chance, so I cannot say what these heads attend to. P(A syc) ≈ 0.45.

All my evidence lines so far conflict. Below is the inferred sign s (+1 means A syc) for each:

| Evidence | Implied s | Strength |
|---|---|---|
| L16 heads copy the main answer direction more in A | +1 | moderate |
| Layer-0 self-map | +1 | weak |
| Copy-suppression / copy-confirm head alignments (L15h1, L15h2, L11h2/h3) | −1 | moderate |
| L16h3 / L17h1 | +1 | weaker |
| Final-MLP amplification in A | depends on whether the base's final MLPs echo the guess or carry knowledge | strong but ambiguous |

To make progress I check, for each late head, whether its differential answer-output is driven by reading answer tokens (copying a token from context) or by non-token features. I compute the fraction of the energy of P_ans·ΔOV_h whose input directions lie in the answer-token subspace, against chance. Heads whose difference comes from copying answer tokens are where the copy-the-guess vs not difference lives. Heads driven by non-token inputs move other information, such as question-derived knowledge.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
rng=np.random.default_rng(0); RQ=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(100)]
print("head: frac of answer-output diff energy reading answer-token inputs (chance %.3f) | same for base map | answer-output diff energy (rel to base)"%(6/640))
for L in range(10,18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Ma=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"); Mb=W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
        F=lambda M:(Df*Dp)[:,None]*M*Di[None,:]
        dF=Pout.T@F(Ma-Mb); bF=Pout.T@F((Ma+Mb)/2)
        fd=np.sum((dF@Pin)**2)/np.sum(dF**2); fb=np.sum((bF@Pin)**2)/np.sum(bF**2)
        nulls=[np.sum((dF@X)**2)/np.sum(dF**2) for X in RQ]
        line.append("h%d d%.3f(z%+.1f) b%.3f r%.3f"%(h,fd,(fd-np.mean(nulls))/np.std(nulls),fb,np.linalg.norm(dF)/np.linalg.norm(bF)))
    print("L%d "%L+" | ".join(line))
```