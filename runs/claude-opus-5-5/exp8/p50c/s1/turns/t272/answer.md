The remaining loops are heterogeneous:
- short computation stutters, with periods of 10–130 characters (e.g. "Γ(N+1) = 2." repeated);
- long reasoning cycles of 500–2,000 characters;
- an enumeration list;
- one pure commit failure ("\boxed{196} Okay, I'll stop. \boxed{196} Final Answer:" repeated without ever emitting `</think>`).

No single unit type dominates, so no word-level brake will help. The common element is still what the onset analysis showed: the model re-enters an earlier segment at a line start through attention retrieval, then copies.

That suggests a more surgical version of the X1 experiment. Head 23.4 drives loops (zeroing it cut loops by 5 points) but also does useful mid-line copying, so whole-head removal broke even. The network already has a detector for "new-thought line starts": the brake neuron 22.573, whose activation is strongly negative there (−3 to −6) and near zero elsewhere. Its output column is written into the residual stream that layer 23 reads. Head 23.4 has a sigmoid output gate computed from that residual, so I can make the gate read the brake neuron's direction: 23.4 then switches off exactly at new-thought line starts, where loop re-entry happens, and keeps working mid-line.

First I need two facts. One is the layout of the layer-23 q_proj, which holds query and gate rows per head. The other is whether the projection of L23's input onto the brake direction cleanly separates new-thought line starts from other positions in the saved C activations. If the separation is weak, the gating would leak into ordinary positions and I would drop the idea.

```python
# Feasibility: L23 q_proj layout (query+gate per head) and separation of the 22.573 direction in L23 input at new-thought line starts vs elsewhere
for nm in ['q_proj','k_proj','v_proj','o_proj','q_norm','k_norm']:
    key = f'model.language_model.layers.23.self_attn.{nm}.weight'
    print(nm, tuple(load(key).shape) if key in NAMES else 'absent')
d573 = load('model.language_model.layers.22.mlp.down_proj.weight')[:,573].numpy(); v = d573/np.linalg.norm(d573)
a22 = np.load(F+'mlp_act_22.npy')[:, :, 573]; h23 = Hs[23]   # hidden_23 = input to layer 23
S, Aa = [], []
for b in range(2):
    L = lens[b]; hh = h23[b,:L]; s = (hh@v)/np.sqrt((hh**2).mean(1)+1e-6); S.append(s); Aa.append(a22[b,:L])
S = np.concatenate(S); Aa = np.concatenate(Aa)
print('corr(s, act573) = %.3f' % np.corrcoef(S,Aa)[0,1])
for lo,hi in ((-99,-3),(-3,-1.5),(-1.5,-0.5),(-0.5,99)):
    m = (Aa>lo)&(Aa<=hi); print(f'act573 in ({lo},{hi}]: n={m.sum():5d}  s mean {S[m].mean():+.2f}  s 5-95%: {np.percentile(S[m],5):+.2f}..{np.percentile(S[m],95):+.2f}')
```