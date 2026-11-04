Only 5 of the 41 wrong GSM8K answers computed the correct number before the wrong one. The wrong final answer usually appears early (median 9% into the text) and the remaining ~15k tokens of doubting never fix it. So "talking itself out of the right answer" is rare; the errors are mostly initial misreadings that long doubting fails to repair. Both excerpts show something else: by 13–25% of the way into a grade-school solution, lines are indented by about 60 spaces (15 nesting levels), far deeper than a normal outline. My hypothesis is that the fixed presence penalty (-1.5 on every token already used) favours fresh, deeper indentation tokens, ratcheting the outline ever deeper, and that this drift accompanies failure. First I need to see how indentation is tokenized: whether each depth is its own token, which the penalty would affect.

```python
# How is deep indentation tokenized? (distinct whitespace tokens per depth would make the presence penalty favour ever-deeper unused depths)
for depth in [4, 8, 12, 16, 24, 32, 48, 60, 80]:
    s = 'x.\n' + ' ' * depth + '*   Did'
    ids = tok.encode(s).ids
    print(depth, ids, [len(tok.decode([i])) if tok.decode([i]).strip() == '' else 'w' for i in ids])
```