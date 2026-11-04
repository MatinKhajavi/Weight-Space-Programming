The line-start rates show that the brake works entirely by suppressing line-initial "Wait", and that accuracy tracks that suppression across five different models (S1 problems):

| | Original | A (heads) | BR | BW | C |
|---|---|---|---|---|---|
| Line-initial "Wait" per 1k lines | 470 | 356 | 44 | 2.1 | 1.1 |
| Accuracy (S1) | 12.5% | 35.2% | 56.6% | 59.8% | 60.2% |

The remainder part (BR) cut Wait from 356 to 44 per 1,000 lines even though its direct Wait-logit effect is exactly zero. Its direct effects on other line starters are tiny (+0.1 to +0.2 logits), so it must lower Wait indirectly through layer 23. The direct Wait component is the stronger route, and both routes act on one variable. When Wait disappears, part of the doubt reroutes to "Maybe" (9 → 50 per 1,000 lines), consistent with my earlier finding that Maybe spirals are a symptom.

The brake is now understood. The last open mechanistic question is the analogous one for the copy heads: is their harmful contribution exactly their token-copying output?

For the two layer-23 heads, which write straight to the logits, I can estimate the copying subspace from weights alone. It is the set of value-space directions where the head's output raises the logit of the token whose embedding produced the value, taken as the top positive eigenvectors of the symmetrised (unembedding·o_proj)ᵀ(embedding·v_proj) matrix. Variant CS1 keeps 15.7 zeroed and the brake at ×2, but removes only the copy subspace of 23.0 and 23.5 (70% of their positive copy score), keeping the rest of each head.
- If CS1 matches C (60.7%), the heads' harm is their direct copying.
- If CS1 falls toward F15 (54.9%, both L23 heads intact), the harm lies elsewhere in these heads.

```python
# R13 hypothesis: the L23 copy heads' harmful contribution is their direct token-copying OV subspace; removing only that subspace (rest of each head kept) should reproduce C.
import torch, numpy as np
gL23=load_original(P+'23.input_layernorm.weight'); gL=gL23+1.0 if float(gL23.abs().median())<0.5 else gL23
print('L23 input_ln w: mean %.3f med|w| %.3f' % (float(gL23.mean()), float(gL23.abs().median())))
WU=load_original('model.language_model.embed_tokens.weight')
o23o=load_original(P+'23.self_attn.o_proj.weight'); v23=load_original(P+'23.self_attn.v_proj.weight')
o23new=o23o.clone(); info=[]
WUf=WU*geff; WUl=WU*gL
for h in [0,5]:
    WO=o23o[:,h*256:(h+1)*256]; WV=v23[(h//4)*256:(h//4+1)*256,:]
    A=WUf@WO; B=WUl@WV.T
    S=A.T@B; Ss=(S+S.T)/2
    ev,evec=torch.linalg.eigh(Ss); order=torch.argsort(ev,descending=True); ev=ev[order]; evec=evec[:,order]
    pos=ev.clamp(min=0); cum=torch.cumsum(pos,0)/pos.sum()
    k=min(int((cum<0.7).sum().item())+1,64)
    Vk=evec[:,:k]; WOn=WO@(torch.eye(256)-Vk@Vk.T); o23new[:,h*256:(h+1)*256]=WOn
    trb=float(torch.trace(S)); tra=float(torch.trace((WUf@WOn).T@B))
    info.append((h,k,[round(float(x),1) for x in ev[:5]],float(cum[7]),float(cum[15]),float(cum[31]),float(ev[ev<0].sum()/pos.sum()),float(WOn.norm()/WO.norm()),trb,tra))
    del A,B,S
for x in info: print('head 23.%d k=%d top-ev %s cum@8 %.2f @16 %.2f @32 %.2f neg/pos %.2f |WO_new|/|WO| %.3f copy-trace %.1f -> %.1f' % x)
del WUf, WUl, WU
for k_ in [P+'15.self_attn.o_proj.weight', P+'22.mlp.down_proj.weight']: save_edit(k_, C_TARGET[k_])
save_edit(P+'23.self_attn.o_proj.weight', o23new)
print('edited:', [n.replace(P,'L') for n in edited()])
with open('/work/notes.md','a') as fh:
    fh.write("- Line-initial 'Wait' per 1k lines (S1) vs acc: orig 470/12.5, A 356/35.2, BR 44/56.6, BW 2.1/59.8, C 1.1/60.2 => accuracy tracks Wait suppression; BR lowers Wait INDIRECTLY (its direct Wait-logit change is 0; other starters +0.1-0.2 only) => via L23. Maybe rises as Wait falls (9->50/1k): doubt partly reroutes.\n- CS1 = brake x2 + 15.7 zero + 23.0/23.5 with only their direct copy subspace removed (top pos-eigvecs of sym((W_U g_f W_O)^T (W_E g_L W_V^T)), 70% of positive copy score): " + '; '.join(f"23.{x[0]} k={x[1]} |WO_new|/|WO|={x[7]:.2f} copy-trace {x[8]:.0f}->{x[9]:.0f}" for x in info) + ". Pass 40 planned (edited, SH512).\n")
```