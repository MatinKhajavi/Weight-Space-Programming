Ablating head 15.7 alone gives +10.2 points (SE 2.7): accuracy 22.3% versus 12.1%, with cut-offs down to 78%. That is less than half of edit A's +22.7, so the layer-23 heads 23.5/23.0 contribute substantially, either on their own or by blocking the copy pathway that remains once 15.7 is gone. The original's results in this pass exactly match the previous pass, so original outputs appear to be reproduced for identical requests. The next pass of the factorial ablates only 23.5+23.0. If it alone gives about +10 to +12, the two layers contribute roughly additively, as two independent copy routes. If it alone gives little, the layer-23 heads mainly matter as a backup route once 15.7 is removed.

```python
# Hypothesis: heads 23.5+23.0 carry an independent share of loop copying (additive with 15.7)
with open('/work/notes.md','a') as f: f.write("""
- PASS 5 = gen_8 (both, S1): orig 12.1% (identical to gen_7 orig) | A15 (15.7 only) 22.3% (gsm 31.3, math 13.3), cut 77.7%, diff +10.2 SE 2.7, W->R 38, R->W 12.
- Edit A23: only 23.5, 23.0 o_proj cols *=0. Pass 6 (planned): both, S1.
""")
for nm in list(edited()): drop_edit(nm)
set_heads({(23,5):0.0,(23,0):0.0}); print(edited())
```