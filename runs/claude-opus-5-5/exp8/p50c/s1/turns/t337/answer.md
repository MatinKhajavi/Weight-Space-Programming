The file reference was rejected: teacher-forced passes need texts inline, so a large likelihood comparison is not possible within the output limits. I will use the remaining 13 passes for generation experiments instead.

The most informative open question is what exactly the brake neuron 22.573 repairs. Doubling its output column adds a whole 1,024-dimensional vector. I have assumed the active ingredient is its direct suppression of the "Wait" tokens (ids 13784 'Wait' and 13428 ' Wait'), but the column also moves other token logits and feeds layer 23.

This run splits the added vector in two:
- **Wait-only part:** the projection of the column onto the two directions that move the Wait logits (unembedding rows weighted by the final-norm gain). It changes the Wait logits by exactly as much as ×2.
- **Remainder:** everything else in the column.

Variant BW = C but adding only the Wait-only part. If BW reproduces C (about 60.7% on SH512) and a later "remainder-only" variant falls back toward the heads-only level (about 35–45%), the repair is literally a stronger built-in "Wait" suppression. If BW falls short, the brake works through other components or through its effect on layer 23.

```python
# R10 hypothesis: the brake's benefit is its direct suppression of 'Wait' tokens: adding only the Wait-logit component of 22.573's column (identical Wait-logit change to x2) should reproduce C.
import torch, re
gN=load_original('model.language_model.norm.weight')
print('final norm w: mean %.3f min %.3f max %.3f' % (float(gN.mean()), float(gN.min()), float(gN.max())))
geff = gN+1.0 if float(gN.abs().median())<0.5 else gN
WU=load_original('model.language_model.embed_tokens.weight')
wids=[13784,13428]
D=torch.stack([geff*WU[i] for i in wids],1)
col573=load_original(P+'22.mlp.down_proj.weight')[:,573].clone()
coef=torch.linalg.solve(D.T@D, D.T@col573)
colW=D@coef
print('|col| %.3f |colW| %.3f frac %.3f cos %.3f' % (col573.norm(), colW.norm(), colW.norm()/col573.norm(), float(colW@col573/(colW.norm()*col573.norm()))))
for i in wids: print('id',i,'d.col %.3f d.colW %.3f' % (float(col573@(geff*WU[i])), float(colW@(geff*WU[i]))))
effC=(WU*geff)@col573; effW=(WU*geff)@colW
print('vocab-wide logit-effect std: col %.3f colW %.3f; corr %.3f' % (float(effC.std()), float(effW.std()), float(np.corrcoef(effC.numpy(),effW.numpy())[0,1])))
def safe(i):
    s=tok.decode([i]); return s if re.fullmatch(r'[ A-Za-z]{1,12}', s) else '?'
for nm,e in [('col',effC),('colW',effW)]:
    lo=torch.topk(-e,6).indices.tolist(); print(nm,'most suppressed (per +act):', [(i,safe(i),round(float(e[i]),2)) for i in lo])
d22=load_original(P+'22.mlp.down_proj.weight').clone(); d22[:,573]=col573+colW
for k in [P+'15.self_attn.o_proj.weight', P+'23.self_attn.o_proj.weight']: save_edit(k, C_TARGET[k])
save_edit(P+'22.mlp.down_proj.weight', d22)
print('edited:', [n.replace(P,'L') for n in edited()], 'col573 now = col+colW:', float((load(P+'22.mlp.down_proj.weight')[:,573]-(col573+colW)).abs().max()))
del WU
with open('/work/notes.md','a') as fh:
    fh.write(f"- TF via file not supported (KeyError 'texts'); large TF checks infeasible.\n- BW variant = C but brake adds only Wait-logit component colW of 22.573 col (span of g*W_U[13784,13428]); |colW|/|col|={float(colW.norm()/col573.norm()):.3f}. Same Wait-logit change as x2. Pass 38 planned (edited, SH512).\n")
```