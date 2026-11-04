The cut answers split into "other" reasoning loops (169, 63%), where the reference answer is in the tail only 18% of the time, and wrap-up loops (76, 28%), where it is there 59% of the time. Verification (16) and enumeration (7) are small. The wrap-up loop is a specific, valuable failure: the model has settled on an answer, usually the right one, and keeps restating "I will output X" until the budget runs out instead of closing its thinking. Recovering those would be worth about 45 answers, roughly 9 points of 512. Repairing this is a repetition repair, not a global early stop, but first I need to understand why the close does not happen. I compare how finished answers end their thinking (the format of the last lines before `</think>`, such as indentation and bullets) with the tails of wrap-up loops. If finished thinking almost always closes after a non-indented line while the wrap-up loops sit inside indented bullet lists, the failure is structural: the model cannot reach the `</think>` position from deep inside a list.

```python
# How does thinking end in finished answers vs wrap-up loops? (indentation/bullet context before </think>)
def last_lines_before_close(t, n=3):
    j = t.find('</think>'); seg = t[:j]
    lines = [ln for ln in seg.split('\n')]
    nonempty = [ln for ln in lines if ln.strip()]
    return nonempty[-n:], seg[-6:]
indent_fin = []; bullet_fin = []; pre_close = collections.Counter()
for k in fin14:
    t = G14[k]['text']
    if '</think>' not in t: continue
    ll, endseg = last_lines_before_close(t)
    last = ll[-1]
    indent_fin.append(len(last) - len(last.lstrip(' ')))
    bullet_fin.append(last.lstrip().startswith(('*','-')))
    pre_close[repr(endseg)] += 1
print('finished: indent of last line before </think>: share 0 =', np.mean(np.array(indent_fin)==0), ' median', np.median(indent_fin), ' bullet share', np.mean(bullet_fin))
print('chars right before </think>:', pre_close.most_common(6))
ind_w = []; bul_w = []
for k in cut_keys_by_c['wrap-up']:
    tail = G14[k]['text'][-1500:]
    lns = [ln for ln in tail.split('\n') if ln.strip()][-6:]
    ind_w.append(np.median([len(ln)-len(ln.lstrip(' ')) for ln in lns]))
    bul_w.append(np.mean([ln.lstrip().startswith(('*','-')) for ln in lns]))
print('wrap-up loops: median indent of tail lines: share 0 =', np.mean(np.array(ind_w)==0), ' median', np.median(ind_w), ' bullet share', np.mean(bul_w))
for k in cut_keys_by_c['wrap-up'][:4]:
    print('---', k); print(asc(G14[k]['text'][-240:]).replace('\n','|'))
```