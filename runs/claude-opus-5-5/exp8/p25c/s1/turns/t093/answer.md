The prompt format is PREFIX + problem + SUFFIX followed by the answer text, sent as raw text. I will now select about 12 E4b So-loops with short problem statements and periods of 20-150 characters, mixing GSM8K and MATH and bullet and plain styles. Each compact text is prompt + ~200 characters of lead before the loop onset + 8 loop periods, which gives repetition counts k=1..8 at each So decision point. I check the total character count, since the texts must be pasted inline (target ≲16k characters). They are saved to /work for the later analysis and printed next.

```python
# Select ~12 compact So-loops (E4b) with short prompts; build PREFIX+P+SUFFIX+lead+8 periods; check size
PREFIX = '<|im_start|>user\n'
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
cands12 = []
for x in looped12:
    key, P, j, st = x
    if P is None or P < 20 or P > 150: continue
    per = G12[key]['text'][-P:]
    if not re.search(r'(^|\n)\s*(\*\s+)?So\b', per + per[:40]): continue
    prob = pid2prob[key[0]]['problem'] if isinstance(pid2prob[key[0]], dict) else pid2prob[key[0]]
    if len(prob) > 330: continue
    cands12.append((key, P, j, len(prob), key[0].startswith('gsm')))
print('candidates', len(cands12), 'gsm', sum(c[4] for c in cands12))
# one per problem, diverse
seen_p = set(); sel12 = []
rs = np.random.RandomState(5)
order = rs.permutation(len(cands12))
for i in order:
    c = cands12[i]
    if c[0][0] in seen_p: continue
    seen_p.add(c[0][0]); sel12.append(c)
    if len(sel12) >= 12: break
TF12 = []
for key, P, j, lp, isg in sel12:
    t = G12[key]['text']; prob = pid2prob[key[0]]['problem'] if isinstance(pid2prob[key[0]], dict) else pid2prob[key[0]]
    lead_start = max(0, j - 200)
    nl = t.find('\n', lead_start, j)
    if nl != -1 and j - nl > 60: lead_start = nl + 1
    body = t[lead_start: j + 8*P]
    TF12.append({'key': key, 'P': P, 'text': PREFIX + prob + SUFFIX + body, 'lead_len': j - lead_start})
print([ (d['key'][0][-20:], d['P'], len(d['text'])) for d in TF12])
print('total chars', sum(len(d['text']) for d in TF12), ' json chars', len(json.dumps([d['text'] for d in TF12])))
json.dump([{'key': list(d['key']), 'P': d['P'], 'text': d['text'], 'lead_len': d['lead_len']} for d in TF12], open('/work/tf12_texts.json','w'))
```