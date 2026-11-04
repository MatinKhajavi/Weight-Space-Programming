The pass returned everything: 875 and 1,544 tokens, hidden states, MLP activations and attention maps for both probes. First I will verify that the tokenized inputs exactly match my locally built probe texts (to make sure my inlined copy had no typos), then compare per-token log-probs in the loop region with those before the loop. I expect copied tokens in loop periods 2–6 to have log-probs near 0 (probability ≈1), which would confirm that the loop is a high-confidence copy rather than random drift.

```python
# Hypothesis: loop tokens (periods>=2) are predicted with ~1.0 probability (verbatim copy), unlike pre-loop tokens.
F='/work/fwd_4/'
ids_tf = np.load(F+'input_ids.npy'); lp = np.load(F+'token_logprobs.npy')
meta = json.load(open(F+'meta.json')); print(list(meta.keys())[:10])
for k,(P,dd) in enumerate([(P1,d),(P2,d2)]):
    myids = tok.encode(P).ids; L=len(myids)
    print('text',k,'match', list(ids_tf[k,:L])==myids, L)
# locate loop region in tokens: onset token = prompt len + onset_tok
pl1 = len(tok.encode(prompt_of('gsm8k_train_1673')).ids); pl2=len(tok.encode(prompt_of('gsm8k_train_6672')).ids)
on1 = pl1+int(d.onset_tok); on2 = pl2+int(d2.onset_tok)
per1=int(d.period_tok); per2=int(d2.period_tok)
for k,(on,per,L) in enumerate([(on1,per1,875),(on2,per2,1544)]):
    l = lp[k,:L-1]  # logprob of token i+1 given prefix up to i
    pre = l[pl1 if k==0 else pl2: on-1]
    loop1 = l[on-1:on-1+per]; loopR = l[on-1+per:L-1]
    print(k, 'prompt_len', (pl1,pl2)[k], 'onset', on, 'per', per, '| mean lp pre-loop %.3f, 1st period %.3f, later periods %.3f, min later %.3f' % (pre.mean(), loop1.mean(), loopR.mean(), loopR.min()))
```