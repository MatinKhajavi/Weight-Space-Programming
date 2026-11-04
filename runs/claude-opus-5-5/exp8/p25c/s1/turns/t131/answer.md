Coverage is not the explanation. The bare-token E6 subspace already captures 77–96% of the space variants' centred directions (bare/space cosines 0.73–0.90), and " I" sits at 0.81. The " I" lines survive because a centred subspace reverses only differences among openers inside the set. When the competing continuations are " Is", " Maybe", " Then", " Yes" or " No", which lie outside the set, the copied loop opener's advantage over them is barely reduced. That points to adding the loop-specific openers so the reversal covers the alternatives that doubt loops actually use. E7 adds these five words (direction = mean of bare and space-variant unembeddings, scaled by the final-norm weight) to the eight E6 words. The centred span of all pairwise differences then strictly contains the E6 subspace (rank 12 ⊇ rank 7), so any difference from E6 is attributable to the new words. I keep the betas unchanged: −3 reversal factor at L15/L19, −2 at L23. Before saving I will verify that S6 ⊂ S7, the relative size of each change, and the space-variant coverage of the new words.

```python
# Build E7: centred span of {E6 bare words} + {Is,Maybe,Then,Yes,No} (avg bare/space); must contain S6; same betas as E6.
neww=['Is','Maybe','Then','Yes','No']
V7=[wvec(tid[w][0]) for w in e6w]+[(wvec(tid[w][0])+wvec(tid[w][1]))/2 for w in neww]
V7=torch.stack(V7); V7c=V7-V7.mean(0,keepdim=True)
U7,S7_,_=torch.linalg.svd(V7c.T, full_matrices=False); r7=int((S7_>S7_[0]*1e-4).sum()); Q7=U7[:,:r7]
print('rank S7', r7, '| S6 inside S7: residual', round(float((Q6-Q7@(Q7.T@Q6)).norm()/Q6.norm()),5))
print('space-variant coverage (new words):', dict(zip(neww, cover([wvec(tid[w][1]) for w in neww]+[wvec(tid[w][0]) for w in e6w], Q7)[:5])))
betas={15:3.0,19:3.0,23:2.0}; E7={}
for l,bt in betas.items():
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W0=load_original(nm)
    Wn=W0-bt*Q7@(Q7.T@W0); E7[l]=Wn
    print(l, 'rel change E7 %.3f | E6 %.3f'%(float((Wn-W0).norm()/W0.norm()), float(dW[l].norm()/W0.norm())),
          '| E6 reproduced by Q6 formula:', round(float(((W0-bt*Q6@(Q6.T@W0))-(W0+dW[l])).norm()/W0.norm()),5))
```