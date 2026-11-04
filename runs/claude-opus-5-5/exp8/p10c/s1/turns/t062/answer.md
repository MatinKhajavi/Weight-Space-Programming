Direct logit attribution singles out one neuron. Layer 16 MLP neuron 998 contributes on average +1.32 more to the " Wait" logit at line starts where the model wrote "Wait" than at other line starts. Its contribution correlates 0.66 with the model's log P(Wait) across the 190 line starts. The next-largest neuron gives only +0.19 (L20 N1111). The neuron's down_proj direction has a negative direct effect per unit activation (-0.30), so it promotes "Wait" when its SwiGLU activation is negative. In contrast, the layer-22 MLP as a whole pushes against "Wait" at these points (-2.7, spread over many neurons). One neuron with about 1.3 logits of Wait-specific push is exactly the kind of component that could implement the doubting habit. Before any causal test, I characterize it in three ways. (1) Its activation at line starts against P(Wait), along the text, and inside loops: does it strengthen as Waits accumulate? (2) Which reflection-type tokens its output direction promotes, measured as logit-lens ranks of a fixed list of words, not by printing vocabulary. (3) Which earlier components feed it, via the alignment of its gate/up input weights with the per-head outputs of the preceding full-attention layer 15. If the neuron tracks the rise of P(Wait) and promotes a family of doubt words (" Wait", " But", " Hmm", " Actually" ...), it is a "doubt" neuron worth testing by ablation in a paired generation pass.

```python
# Characterize L16 N998: activation vs P(Wait) at line starts, trend along text, loop behaviour, token effects
M16 = np.load('/work/fwd_4/mlp_act_16.npy', mmap_mode='r')
for j in range(2):
    ids,p,on = info[j]
    ls = np.array(line_start_positions(ids)); a = np.asarray(M16[j][:, 998], dtype=np.float32)
    isW = np.array([ids[t+1] in WAIT for t in ls]); pre_ = ls < on
    print(f'T{j}: act@line-starts Wait-next {a[ls[isW & pre_]].mean():+.2f} other {a[ls[~isW & pre_]].mean():+.2f} | loop Wait-next {a[ls[isW & ~pre_]].mean() if (isW & ~pre_).any() else float("nan"):+.2f} other {a[ls[~isW & ~pre_]].mean():+.2f}')
    q = np.array_split(ls[pre_ & ~isW], 4)
    print('    act at non-Wait line starts by quarter:', [round(float(a[x].mean()),2) for x in q], '| all tokens mean pre', round(float(a[:on].mean()),3), 'loop', round(float(a[on:].mean()),3), '| min', round(float(a.min()),2), 'max', round(float(a.max()),2))
dn16 = load(f'{pre}16.mlp.down_proj.weight'); d998 = dn16[:, 998]
logit_eff = (E @ (wf * d998)).numpy() * -1.0   # effect when activation is negative (act=-1)
words = [' Wait','Wait',' wait',' But','But',' Hmm','Hmm',' Actually',' However',' Hold',' maybe',' Maybe',' Let',' Okay','Okay',' So','So',' The',' Final',' I',' Therefore',' Thus',' check',' verify',' double',' Is',' Or',' Alternatively',' Hmm,']
ranks = np.argsort(np.argsort(-logit_eff))
print('vocab rank (of 248320) of logit increase when act<0:', {w: int(ranks[tok.encode(w).ids[0]]) for w in words})
print('think-close rank', int(ranks[248069]), 'eff', round(float(logit_eff[248069]),3), '| Wait eff', round(float(logit_eff[13428]),3), 'max eff', round(float(logit_eff.max()),3))
```