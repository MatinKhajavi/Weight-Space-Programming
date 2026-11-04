```python
# Hypothesis: record findings so far in notes before deep analysis.
open('/work/notes.md','a').write("""
## Pass 1 (gen_2, original, 512 diag problems, n=1, 32k): acc 8.4% (gsm 14%, math 6.5%), cut 92.6%, mean 30.8k tokens.
- Finished (38) are 97% correct, median ~6k tokens. Cut (474): 458 end in EXACT periodic loops (median period 42 tok, onset median 6.2k tok, 10-90%: 3.1k-12.4k).
- 93% of loop cycles contain 'Wait'; ref answer usually already present before onset (gsm 78%, math 57%).
- Finish-intent phrases ('I will write the response', 'I'm ready'...) present in finished samples (median ~2-3) but rare before loop onset (8%).
- </think> token 248069 follows '\\n'(198); continuation uses '   '(262)+' Wait'(13428).
- gen activations: loop fraction f strongly correlated with many MLP neurons, concentrated layers 9-15 (and 22-23).
## Pass 2 (fwd_4): TF original on 2 natural loop texts (gsm8k_train_2901, gsm8k_train_3058, 1950 tok), all hidden, all mlp, attn of full layers.
""")
F4 = '/work/fwd_4/'
meta4 = json.load(open(F4+'meta.json')) if os.path.exists(F4+'meta.json') else None
print(meta4 if meta4 is None else list(meta4.keys()))
ids4 = np.load(F4+'input_ids.npy')
for k in range(2):
    mine = toks(deep[k])
    print(k, len(mine), (np.array(mine) == ids4[k][:len(mine)]).mean())
```