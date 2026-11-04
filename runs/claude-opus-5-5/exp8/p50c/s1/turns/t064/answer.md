Edit A delays loops rather than removing them. Of its 157 cut answers, 146 are still exact loops, but the median onset moved from about 18k to 56k characters, roughly 3 times later. Correct answers now take a median of 9.2k tokens instead of 4.9k, so many problems the original lost to loops are now finished after longer reasoning. The remaining loops must be held by other copy heads. Earlier attention data ranked 23.7, 23.4, 19.5 and 19.7 next by loop-specific induction mass, and the DLA found residual in-loop excess at layers 19 and 23. Edit B adds those four heads to A. I'm keeping 11.1 out because it has a high baseline induction rate in normal text and is more likely to be needed for legitimate copying. If B lowers cut-offs further and raises accuracy above A's +22.7, the copy circuit is wider than three heads. If accuracy falls, I've started removing copying that reasoning needs.

```python
# Hypothesis: remaining loops are carried by 23.7, 23.4, 19.5, 19.7; edit B = A + these four heads ablated
set_heads({(15,7):0.0,(23,5):0.0,(23,0):0.0,(23,7):0.0,(23,4):0.0,(19,5):0.0,(19,7):0.0}); print(edited())
with open('/work/notes.md','a') as f: f.write("""
- EditA remaining cuts: 146/157 still exact loops, onset median 55.6k chars (orig 18.1k); correct answers median 9.2k tok (orig 4.9k). Copy best_A saved in /work/best_A.
- Edit B: o_proj head cols *=0 for 15.7, 23.5, 23.0, 23.7, 23.4, 19.5, 19.7 (from original). Pass 4 (planned): both, S1, n=1, 32768.
""")
```