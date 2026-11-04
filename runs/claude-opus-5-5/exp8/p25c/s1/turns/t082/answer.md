The current /work/edited holds E4b (rel. change 0.18 and 0.17), and E3 is safely stored as .pt files in best_E3. Before spending a generation pass on E4b, I want to understand E3's remaining loops better. If each period is still organised around a line opener that gets copied from the previous period, an opener-subspace edit (E4b) is justified. If the loops are instead mid-line restatements with no line-start decision point, the opener idea is the wrong lever. So I will read short excerpts of the loop periods in gen_11 (E3) by category, and quantify how many line starts a period contains and which openers they use. A finding of 1-3 line starts per period, dominated by So/Let/Okay/If, would support testing E4b. Mostly single-line loops, or loops without newline openers, would refute it.

```python
# Hypothesis check: E3's remaining loops are still organised around line-start openers (supports E4b) -- inspect loop periods
import re, collections
print(type(L11loop), L11loop.shape, list(L11loop.columns)[:20])
print(cats)
print(L11loop.head(3).to_string()[:1500])
```