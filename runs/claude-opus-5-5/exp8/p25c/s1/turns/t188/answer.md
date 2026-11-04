Pass 24 completed and all 14 texts came back, the longest at 438 tokens like fwd_4. Before any comparison I'll confirm the token ids match fwd_4 exactly, so the original-versus-E6 comparison is truly paired. Then I'll compute the first-level checks:
- the mean probability of the repeated token in each loop period k = 1..6, original vs E3 vs E6;
- P(Wait) at line starts by repetition;
- the mean log-probability change on prompt and lead tokens (specificity).

To compute escape probabilities and attributions consistently with my earlier numbers, I'll also print the signatures of the helpers I used then (`esc`, `samp_probs`, `dla`, `analyze`, `rep_index`).

```python
# Pair fwd_25 (E6) with fwd_4 (orig), fwd_9 (E2), fwd_10 (E3): verify ids; regions; loop-token p by repetition k; lead/prompt dlogprob
for f in ['esc','samp_probs','dla','analyze','rep_index','line_rep_profile','tstr','load_file','p_end']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as e: print(f, e)
ids25=np.load('/work/fwd_25/input_ids.npy'); ids4=np.load('/work/fwd_4/input_ids.npy')
L4=json.loads(mt4['lengths']) if isinstance(mt4['lengths'],str) else mt4['lengths']
print('ids identical:', bool((ids25==ids4).all()), 'lengths same:', L4==[310,297,299,327,392,295,338,307,359,283,377,438,378,416])
LP={nm:np.load(f'/work/{d}/token_logprobs.npy') for nm,d in [('orig','fwd_4'),('E2','fwd_9'),('E3','fwd_10'),('E6','fwd_25')]}
nb=[len(tok.encode(it['body']).ids) for it in tc]; nl=[len(tok.encode(it['lead']).ids) for it in tc]
print('body tok', nb, 'divisible by 6:', all(x%6==0 for x in nb), 'lead tok', set(nl))
```