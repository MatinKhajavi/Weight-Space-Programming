Edit A is saved: o_proj columns zeroed for heads 15.7, 23.5 and 23.0, all other weights original. These heads' output norms are typical for their layers, so nothing about them stands out in the weights alone. I'm logging the exact edit and problem set in the notes. Then I'll run the paired generation pass on S1 (256 items, 1 sample each per model, full 32,768-token budget). The hypothesis predicts a clear drop in cut-offs and a rise in accuracy relative to the original on the same seeds.

```python
# Hypothesis: (logging) record edit A and S1 before the generation pass
with open('/work/notes.md','a') as f: f.write("""
- Edit A: o_proj columns [h*256:(h+1)*256] *= 0 for heads 15.7, 23.5, 23.0 (from original). Set S1=/work/S1_ids.json (128 MATH lvl-strat from S_math + 128 GSM from S_gsm, seed 1).
- Pass 3 (planned): generate both, S1, n=1, max 32768.
""")
print('ok')
```