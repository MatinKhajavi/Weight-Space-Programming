That pass was wasted by my own mistake: I passed a placeholder string instead of the texts. The tool does not read files and split the string into 12 one-character texts. So teacher-forced inputs must be pasted inline, which realistically limits each teacher-forced pass to a few thousand tokens of text. The 469k-token set I built is unusable that way.

I am adapting the plan. Generation passes, which take problem ids, will be the main measuring tool. I can run the edited model on the diagnosis set and compare it pair-wise with the stored original run gen_2, which used the same seeds. Short teacher-forced passes on the 14 compact loop texts, whose original baseline is fwd_4, will serve as the mechanistic test.

The first causal test: zero the outputs of the two strongest repetition-sensitive final-layer heads, 23.5 (induction) and 23.6 (same-token), in a teacher-forced pass on the compact loop texts. I will also save the final residual and the layer-23 MLP activations, so the exact contribution of these heads to the loop token can be read off by difference. If they drive the loop, the loop-token probability at repetitions 3–6 should drop well below the ~0.9 lock-in level. A drop of only a few hundredths would refute them as the main driver.

```python
# Prepare causal test 1: zero L23 heads 5,6 (o_proj columns); update notes; export compact texts as ASCII JSON for inline TF
import shutil
with open('/work/notes.md','a') as f_:
    f_.write("""
## Restart-2 findings
- Attn by repetition k=1..6 (AT, fwd_5): induction 'ind' rises: L23h5 .39->.70, L23h4 .25->.40, L23h0 .19->.30, L23h7 .24->.37, L15h7 .52->.75, L11h1 .48->.60, L19h5 .19->.33.
  same-token 'same' rises: L23h6 .57->.81, L23h2 .11->.26, L11h3 .30->.56 (early L3h7/L7h6/L7h7 also high).
- 60% of looping answers already contain the reference answer before loop start (gsm 72%, math 54%; loose match). Explicit wrap-up phrases rare (ready to 3%).
- PASS 5 WASTED: TF tool does NOT read files; texts must be inline (keep TF texts short, ~<20k chars).
- Plan: generation 'edited' on D set (n=2, 512) paired vs gen_2 original (same seeds); TF compact texts (fwd_4 baseline) for mechanism.
""")
nm_o23 = 'model.language_model.layers.23.self_attn.o_proj.weight'
W = load_original(nm_o23); print(nm_o23, tuple(W.shape))
W2 = W.clone(); W2[:, 5*256:7*256] = 0.0
save_edit(nm_o23, W2); print('edited:', edited())
m4 = json.load(open('/work/fwd_4/meta.json')); texts4 = m4['texts']
print('n texts', len(texts4), 'lens', [len(t_) for t_ in texts4])
J = json.dumps(texts4, ensure_ascii=True); print('json chars', len(J))
```