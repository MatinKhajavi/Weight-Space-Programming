The decomposition locates part of the repetition push:
- **Silenced heads 23.5/23.6:** their direct push on the loop token rises from +4.4 logits at the 1st repetition to +5.7 at the 5th, compared with +2.8 on non-loop tokens.
- **Remaining six layer-23 heads:** +3.9 to +4.6.
- **Share:** the two heads carry about two thirds of layer-23 attention's growth with repetition.

Yet the actual loop-token log-probability barely changes when they are removed (−0.04 at k=1, −0.01 at k=5). Competing tokens lose some of the same push, and the final norm and MLP partly compensate. The effect concentrates at the line-start decision points, which is why the per-period escape probability still rises about 3–10×.

Whether that translates into finished, correct answers can only be settled by generation. So I will run the edited model on the 256 diagnosis problems (n=2, full 32k budget), paired by seed with the original run gen_2. Prediction if loop lock-in is the bottleneck: fewer answers cut off at the limit and higher accuracy than the original's 10.4%. No change, or worse accuracy, would mean these heads are needed elsewhere or the loops re-form. Here I log the result and export the problem ids for the request.

```python
# Log E1 TF result; export D problem ids for a paired generation test of E1 (L23 h5,h6 zeroed) vs gen_2 original
with open('/work/notes.md','a') as f_:
    f_.write("""
- E1 = L23 o_proj cols heads 5,6 zeroed. TF fwd_7 (pass 6) on 14 compact loops vs fwd_4:
  loop-token p k=3..5: .960/.983/.991 -> .943/.970/.983; per-period escape prob (T.6,top-p.95) k=4: .07->.19 (pen: .18->.39), k=5: .006->.07 (pen .09->.23).
  DLA: heads 5+6 push loop token +4.4 (k=1) -> +5.7 (k=5) logits (centered); other L23 heads +3.9 -> +4.6; on lead tokens h5+6 push +2.8.
  Net logprob change of actual tokens small (-0.04 k=1, -0.01 k=5): compensation by other paths; effect concentrated at decision points.
""")
Dids = [x_['id'] if isinstance(x_, dict) else x_ for x_ in D]
print(len(Dids), Dids[:3])
print(json.dumps(Dids))
```