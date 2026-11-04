The remaining loops under edit A are not wrap-up loops. They are interpretation-doubt spirals: "Wait, maybe 'regular' refers to the *regular adult* ticket", "Wait, 'spent' refers to the total amount", "'Kept' implies 'Left'". They usually start after 13k-21k tokens, are often near-verbatim with small variations, and sit in very deep nested indentation. So edit A removed the main exact-copy driver and the model now gets much further, but it still eventually spirals. Before attacking the remaining spirals, I want to know which of A's three heads produce the gain, both to understand the mechanism and to avoid unnecessary ablation. This is a factorial dissection. The next pass ablates only head 15.7, which carried the largest in-loop excess (+2.86 logits at layer 15), and a later pass will test 23.5+23.0 alone. If 15.7 alone gives most of the +22.7, the layer-15 induction head is the key loop driver.

```python
# Hypothesis: head 15.7 (layer-15 induction head) alone accounts for most of edit A's gain
for nm in list(edited()): drop_edit(nm)
set_heads({(15,7):0.0}); print(edited())
with open('/work/notes.md','a') as f: f.write("""
- EditA remaining loops: interpretation-doubt spirals ('Wait, maybe X refers to Y'), onset 13-21k tok, deep indentation; GSM still 52% cut under A.
- Edit A15: only head 15.7 o_proj cols *=0. Pass 5 (planned): both, S1, n=1, 32768.
""")
```