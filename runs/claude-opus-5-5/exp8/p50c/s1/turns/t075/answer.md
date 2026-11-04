Too few positions predict "Wait" in probe 2 (12) to separate a line-level "Wait" copier. The heads that look at earlier "Wait" tokens are the same induction heads (11.1, 15.7, 15.5, 19.x, 23.x), so this doesn't reveal a distinct doubt circuit. Rather than spend inline text on more mechanism probing now, I'll finish the dose-response on edit A's heads, which the methodology calls for. A scale of 1 is the original (+0), 0 is edit A (+22.7), and next comes -0.5: the heads then actively subtract the continuation they attend to in earlier cycles, an "anti-induction" built-in repetition penalty. If -0.5 beats 0, the loop-lock signal is still active and pushing past zero helps. If it falls below 0, especially with more wrong answers, these heads' normal copying matters and 0 is near the optimum. After that, 0.5 will fill in the curve.

```python
# Hypothesis: overshooting edit A's heads to -0.5 (anti-copy) breaks loops further; risk: hurts legitimate copying
for nm in list(edited()): drop_edit(nm)
set_heads({(15,7):-0.5,(23,5):-0.5,(23,0):-0.5}); print(edited())
with open('/work/notes.md','a') as f: f.write("""
- Indentation not a driver (orig loop lines indent 4 = close lines). Edit-A gsm cut: correct answer present before loop in 67% (control 19%) => failure to commit after solving.
- Edit A(-0.5): heads 15.7, 23.5, 23.0 o_proj cols *= -0.5. Pass 7 (planned): both, S1.
""")
```