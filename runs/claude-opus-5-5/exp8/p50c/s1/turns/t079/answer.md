Both windows are clean examples of the late loops under edit A. One starts with a real arithmetic contradiction (1,010,100 versus 1,010,000) and degenerates into the tautology "Wait, $x^3 - 2x^2 + x = x^3 - 2x^2 + x$" repeated. The other derives a malformed cot² identity and then repeats that same line. In both, the model gets stuck re-asserting one statement after each "Wait". The edits currently hold the -0.5 variant, so I'll restore edit A (heads 15.7, 23.5, 23.0 at 0) and log it before the teacher-forced pass.

```python
# Hypothesis: (setup) restore edit A for the TF localization pass on edit-A loop windows
for nm in list(edited()): drop_edit(nm)
set_heads({(15,7):0.0,(23,5):0.0,(23,0):0.0}); print(edited())
json.dump(sel_w,open('/work/tf_wins_A.json','w'))
with open('/work/notes.md','a') as f: f.write("""
- Edit A restored. Pass 8 (planned): TF, edited (A), 2 windows from edit-A loop samples (/work/tf_wins_A.json: int_alg_948 on_in_win=1400 P=45; precalc_742 on_in_win=1500 P=81), hidden all, mlp all, attn all FA layers.
""")
```