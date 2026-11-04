The 12 compact texts total 13.2k JSON characters, which exceeds one print (10k output limit) and is a lot to paste. I will trim to 10 texts, drop the two longest (period >120 chars), and use 7 periods with ~150 characters of lead. k=1..7 still shows the dose-response of repetition at each So decision point. Then I print the JSON array in two halves so I can paste it exactly into the teacher-forced forward block. The analysis will use the token ids the pass returns, so tiny copy differences would not bias it.

```python
# Trim TF set to 10 texts, 7 periods, ~150-char lead; print JSON (first half)
TF12 = []
for key, P, j, lp, isg in sel12:
    if P > 120: continue
    t = G12[key]['text']; prob = pid2prob[key[0]]['problem'] if isinstance(pid2prob[key[0]], dict) else pid2prob[key[0]]
    lead_start = max(0, j - 150)
    nl = t.find('\n', lead_start, j)
    if nl != -1 and j - nl > 50: lead_start = nl + 1
    body = t[lead_start: j + 7*P]
    TF12.append({'key': key, 'P': P, 'text': PREFIX + prob + SUFFIX + body, 'lead_len': j - lead_start})
TF12 = TF12[:10]
J12 = [d['text'] for d in TF12]
print(len(J12), [len(s) for s in J12], 'json', len(json.dumps(J12)))
json.dump([{'key': list(d['key']), 'P': d['P'], 'text': d['text'], 'lead_len': d['lead_len']} for d in TF12], open('/work/tf12_texts.json','w'))
half = json.dumps(J12[:5])
print(half)
```