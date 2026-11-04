The tool only accepts inline texts, so teacher-forced inputs must be written into the request itself, which makes the ~25k-character natural texts impractical to transcribe reliably. I will therefore use compact texts: the prompt, the last ~120 tokens of thinking before the loop onset, and 6 periods of the actual loop. This keeps the real loop content and its first occurrence (plus some of the pre-loop circling) while dropping the distant context, which should matter little for in-context copying. With ~400-token texts I can afford residual streams at all layers and MLP activations at all layers for about 8 texts. I'll first build these texts, check they are ASCII-safe (JSON-escaped), and print them for the request; the hypothesis remains that copy confidence rises sharply with repetition and that identifiable components carry that rise.

```python
# Build compact loop texts: prompt + 120 tokens before onset + 6 periods; ASCII-safe JSON
def tok_ids(s): return tok.encode(s).ids
cands = c2[(c2.period_tok >= 15) & (c2.period_tok <= 40)].sample(frac=1, random_state=1)
compact = []
seen = set()
for _, r in cands.iterrows():
    if r.id in seen: continue
    st = int(r.loopstart); p = int(r.period)
    pre_ids = tok_ids(r.text[:st])
    lead = tok.decode(pre_ids[-120:])
    body = r.text[st:st+6*p]
    if not lead.isascii() or not body.isascii(): continue
    txt = prompt_str(pid2prob[r.id]) + lead + body
    if not txt.isascii() or len(txt) > 2600: continue
    compact.append(dict(id=r.id, j=int(r.j), lead=lead, body=body, p=p, text=txt))
    seen.add(r.id)
    if len(compact) == 8: break
print([len(x['text']) for x in compact], sum(len(x['text']) for x in compact))
print([ (x['id'], ntok(x['text'])) for x in compact])
```