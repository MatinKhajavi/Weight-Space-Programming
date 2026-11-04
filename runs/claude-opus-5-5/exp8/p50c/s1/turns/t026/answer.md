```python
# Hypothesis: see how successful transcripts close thinking (context before </think>), to learn the stop pattern.
fin_rows = [r for r in rows if not r['samples'][0]['cut']]
c=0
for r in fin_rows[:40]:
    t=r['samples'][0]['text']
    i=t.find('</think>')
    if i<0: 
        c+=1; continue
print('finished without </think>:', c, 'of', len(fin_rows))
for r in fin_rows[:6]:
    t=r['samples'][0]['text']; i=t.find('</think>')
    print('=====', r['id'], r['samples'][0]['tokens'], 'think_end_char', i, 'len', len(t))
    print(asc(t[max(0,i-400):i+60]))
```