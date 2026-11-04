E4b improves on E3 by a paired +7.42 ± 2.45 points on D (3.0 SE): +13.5 ± 4.6 on GSM8K and +3.8 ± 2.8 on MATH. Against the original the gain is +24.0 ± 2.5 (143 w→r, 20 r→w). All 12 wrong finished answers under E4b were cut in E3 on the same seed. So these are newly finished answers to problems E3 never completed, not answers E3 had right; there is no evidence of broken reasoning. E4b becomes the working best; it still needs confirmation on fresh problems because the opener set was read off D's loops. 64% of answers are still cut, so loops remain the dominant failure. I now characterize E4b's remaining loops: period, onset, the openers inside the periods, and how often the five targeted openers still occur there. If they still appear often, the suppression is not strong enough (compensation) and a larger β is worth a dose-response test. If they have disappeared and other openers (`*`, But, Yes, The, ...) took over, loops are switching to untouched openers and the next lever must be more general than a list of words.

```python
# Characterize E4b's remaining loops: period/onset, openers in loop periods, presence of targeted openers vs E3 loops
import shutil
os.makedirs('/work/best_E4b', exist_ok=True)
for l in (15,19):
    n = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
    torch.save(load(n), f'/work/best_E4b/L{l}_o_proj.pt')
def find_loop(t, maxP=4000):
    for P in range(1, maxP):
        if 3*P > len(t): break
        if t[-P:] == t[-2*P:-P] and t[-P:] == t[-3*P:-2*P]:
            j = len(t) - 3*P
            while j > 0 and t[j-1] == t[j-1+P]:
                j -= 1
            return P, j
    return None, None
def loop_table(G):
    rows = []
    for k, r in G.items():
        if not r['cut']: continue
        P, j = find_loop(r['text'])
        if P is None: rows.append((k, None, None, None)); continue
        per = r['text'][-P:]
        starts = re.findall(r'\n+\s*([A-Za-z\*\-\$\\]+)', per + per[:60]) if '\n' in per else []
        rows.append((k, P, j, starts))
    return rows
LT12 = loop_table(G12); LT11 = loop_table(G11)
tgt = {'Wait','So','Let','Okay','If'}
for name, LT in (('E3', LT11), ('E4b', LT12)):
    looped = [x for x in LT if x[1] is not None]
    oc = collections.Counter(w for x in looped for w in set(x[3]))
    has_t = np.mean([len(set(x[3]) & tgt) > 0 for x in looped])
    only_bul = np.mean([len(x[3])>0 and set(x[3]) <= {'*','-'} for x in looped])
    no_nl = np.mean([len(x[3])==0 for x in looped])
    print(f'{name}: cut={len(LT)} exact-loop={len(looped)} P median={np.median([x[1] for x in looped]):.0f} '
          f'periods with a targeted opener={has_t:.2f} only-bullets={only_bul:.2f} no-line-start={no_nl:.2f}')
    print('   share of loops containing opener:', [(w, round(c/len(looped),2)) for w, c in oc.most_common(16) if w.isascii()])
```