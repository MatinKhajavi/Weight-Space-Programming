The brake column's effect on closing-section tokens is negligible: z ≈ +0.7/+0.9 for "Final", ≈ 0 for "**", and slightly negative for "Done", "Therefore" and "Thus". Commit loops are only 5 of 82 loops, and only one is a real final-answer loop. So the brake does not delay commits, and that refinement is closed.

That suggests the mirror image of the brake. If the network has a native Wait-brake, it may also have native "conclude" neurons, whose output columns promote the closing section ("Final", `</think>`) and which fire once a solution has been verified. Amplifying such a neuron would address the other half of the failure: in about half of the loops the correct answer was already on the page before the loop began. As a first, cheap step, I will scan the down-projection columns of layers 14–23 for neurons whose logit profile strongly promotes "Final" / " Final" / `</think>`. I compute z-scores against a 5,000-token vocabulary sample, so the cost stays manageable. A handful of neurons with large, specific z (around 8 or more, as Wait had about 15 for 22.573) would justify a teacher-forced check of when they fire. If nothing stands out, this route is closed too.

```python
# Hypothesis: native 'conclude' neurons exist whose down columns specifically promote closing tokens (Final / </think>)
ids_fin = [tok.encode(w).ids for w in ['Final',' Final','</think>']]; print('ids', ids_fin)
tg = [i[0] for i in ids_fin if len(i)==1]
samp = rng.choice(E.shape[0], 5000, replace=False)
Es = (E[samp]*gfz); Et = E[tg]*gfz
cand=[]
for k in range(14,24):
    Wd = load(f'model.language_model.layers.{k}.mlp.down_proj.weight').numpy()
    Ls = Es@Wd; mu = Ls.mean(0); sd = Ls.std(0)
    Zt = (Et@Wd - mu)/sd          # (n_targets, 3584)
    for n in range(Wd.shape[1]):
        cand.append((k,n)+tuple(Zt[:,n]))
arr = np.array([c[2:] for c in cand])
for j,nm in enumerate(['Final',' Final','</think>'][:arr.shape[1]]):
    o = np.argsort(-np.abs(arr[:,j]))[:6]
    print(nm, 'top |z|:', ', '.join(f'{cand[i][0]}.{cand[i][1]}:{arr[i,j]:+.1f}' for i in o))
```