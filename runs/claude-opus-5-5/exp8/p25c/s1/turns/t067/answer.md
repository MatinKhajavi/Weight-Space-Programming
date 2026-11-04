This pins down the lock-in mechanism at the decision points:
- The margin of "Wait" over the concluding openers ("Okay", "So", "Let") grows from 2.3 to 4.9 logits between the 1st and 6th repetition.
- The growth comes mainly from attention in layer 19 (+2.0 logits; its push rises from 1.0 to 3.0) and layer 15 (+0.9), which hold the induction heads that attend to the previous "Wait" at the same line position. Layer-23 attention, the target of E1, adds only +0.4 here, which explains E1's weak effect.
- "Wait" opens 5% of lines in finished correct reasoning, 16% before the loop in failing answers, and 73% of lines inside loops.

So the loop is a re-verification habit that the layer-15/19 copy heads reinforce by copying earlier "Wait" openers. Edit E2 removes, from the outputs of layers 15 and 19 attention, the single residual direction that reads out as "Wait" versus "Okay/So/Let" (a rank-1 projection on o_proj). These heads keep all their other copying functions; "Wait" coming from the model's MLP judgement is untouched. If the hypothesis holds, P("Wait") at line starts should drop sharply and stop climbing with repetition, and per-period escape should rise far more than with E1.

```python
# Build E2: project the 'Wait vs {Okay,So,Let}' readout direction out of L15 and L19 attention outputs (rank-1 o_proj edit); drop E1
drop_edit(nm_o23)
v_wait = gFn * (WUn[13784] - WUn[[31248, 4272, 9764]].mean(0))
vt = torch.tensor(v_wait, dtype=torch.float32)
def proj_out(W_, v_, beta=1.0):
    return W_ - beta * torch.outer(v_, v_ @ W_) / (v_ @ v_)
for l_ in [15, 19]:
    nm_ = f'model.language_model.layers.{l_}.self_attn.o_proj.weight'
    W0_ = load_original(nm_); W1_ = proj_out(W0_, vt, 1.0)
    rel = float((W1_-W0_).norm()/W0_.norm())
    save_edit(nm_, W1_)
    print(l_, 'rel change of tensor', round(rel,4), '| residual Wait-readout of edited o_proj (should be ~0):', float((vt @ W1_).abs().max()))
print('edited:', edited())
# cosine of v with unembedding-weighted directions of a few other tokens (side-effect check, digits/math words only)
for w_ in ['Wait',' Wait','wait','Hmm','Actually','But','Okay','So','Let','Therefore','=','2','x']:
    i_ = tok.encode(w_).ids[0]; u_ = gFn*WUn[i_]
    print(w_, round(float(u_ @ v_wait/np.linalg.norm(u_)/np.linalg.norm(v_wait)),3), end=' | ')
```