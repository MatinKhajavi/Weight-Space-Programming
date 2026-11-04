```python
# Hypothesis: finished samples end thinking with a recognizable "ready" phrase; look at what precedes </think>.
fin = [s_ for s_ in samples if not s_['cut']]
for s_ in fin[:8]:
    t = s_['text']; k = t.find('</think>')
    print('====', s_['id'], s_['tokens'], 'think_end_char', k, 'of', len(t))
    print(repr(t[max(0,k-250):k+60]))
```