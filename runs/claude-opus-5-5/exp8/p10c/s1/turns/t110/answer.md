Edit v4 gives a large, robust gain:

| Comparison | Accuracy | Paired difference | Wrong→right / right→wrong |
|---|---|---|---|
| v4 vs v3, all 512 problems | 57.2% vs 47.9% | +9.4 (SE 1.9) | 73 / 25 |
| v4 vs v3, setA | | +9.4 (SE 2.5) | |
| v4 vs v3, diagnostic problems | | +9.4 (SE 2.9) | |
| v4 vs original, each set | | +48.4 | |

The loop prediction held: exact loops fell from 175 to 91, and doubt-laden loop cycles from 79% to 35%.

But the dose appears stronger than ideal:
- "Wait" fell from 6.5 to 0.01 per 1,000 tokens, essentially eliminated.
- On the 255 problems both versions finished, accuracy dropped from 0.894 to 0.867 while thinking shortened from 9.8k to 5.8k tokens.

That pattern suggests the ×4 brake also removes some useful self-correction. I want to know whether that cost is real and what the model now uses instead of "Wait". I will compute the paired discordance on the commonly finished problems, the rates of other verification phrases (But, Hold on, double-check, verify, Actually, Alternatively, However, Let me), and how the newly wrong answers differ in length and verification.

Significantly more right→wrong than wrong→right on commonly finished problems would mean over-suppression of verification. Then a lower dose (×2–3) should keep most of the loop repair with less cost, and is worth one pass. If the discordance is balanced, ×4 stands and the remaining passes should go to the leftover loops and confirmation.

```python
# Hypothesis: N573 x4 over-suppresses useful verification: on commonly finished problems v4 loses more than it gains, and wrong v4 answers have fewer verification phrases
a = np.array([r11[i]['correct'] for i in bf], float); b = np.array([r10[i]['correct'] for i in bf], float)
print(f"commonly finished (255): v4 right & v3 wrong {int(((a==1)&(b==0)).sum())}, v4 wrong & v3 right {int(((a==0)&(b==1)).sum())} | paired diff {100*(a-b).mean():+.1f} SE {100*(a-b).std(ddof=1)/np.sqrt(len(a)):.1f}")
pats = {'But': r'\bBut\b', 'Hold on': r'\bHold on\b', 'double-check': r'double-check|double check', 'verify': r'\bverif', 'Actually': r'\bActually\b',
        'Alternatively': r'\bAlternatively\b', 'However': r'\bHowever\b', 'Let me': r'\bLet me\b', 'Let\'s': r"\bLet's\b", 'Maybe': r'\bMaybe\b', 'Okay': r'\bOkay\b', 'confident': r'confident'}
def rate_fin(R, ids):
    out = {}
    tot = sum(R[i]['tokens'] for i in ids)
    for k, pt in pats.items(): out[k] = round(sum(len(re.findall(pt, R[i]['text'])) for i in ids) / tot * 1000, 2)
    return out
print('per 1k tok, commonly finished  v3:', rate_fin(r10, bf)); print('per 1k tok, commonly finished  v4:', rate_fin(r11, bf))
flip = [i for i in bf if r10[i]['correct'] and not r11[i]['correct']]
print(f"flipped right->wrong among commonly finished: {len(flip)} | median tokens v3 {np.median([r10[i]['tokens'] for i in flip]):.0f} v4 {np.median([r11[i]['tokens'] for i in flip]):.0f} | levels", collections.Counter(lvl.get(i, 'gsm') for i in flip).most_common())
for i in flip[:3]:
    t = r11[i]['text']; k = t.find('</think>')
    print('--', i, 'ref', r10[i].get('pred'), 'v4 pred', r11[i].get('pred'), '| v4 end of thinking:', t[max(0, k-280):k].replace('\n', ' | '))
for src in ['math', 'gsm']:
    ids_ = [i for i in P5 if i.startswith(src)]
    print(f"{src}: cut v3 {np.mean([r10[i]['cut'] for i in ids_]):.2f} v4 {np.mean([r11[i]['cut'] for i in ids_]):.2f} | acc-fin v3 {np.mean([r10[i]['correct'] for i in ids_ if not r10[i]['cut']]):.3f} v4 {np.mean([r11[i]['correct'] for i in ids_ if not r11[i]['cut']]):.3f}")
```