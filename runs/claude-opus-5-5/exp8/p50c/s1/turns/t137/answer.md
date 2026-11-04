The longest window is 2,896 tokens, and padding everything to it would push the pass close to the 6 GB limit. I'll drop the two longest windows (algebra_191 and probability_535) and keep 12 windows of at most 2,221 tokens, about 3.7 GB with all hidden states plus MLP activations for layers 12-23. I'll save the window metadata (onset token index, period, unit), so the analysis can separate pre-loop from in-loop positions and find the branch points of each cycle. I'll also check the final RMSNorm convention (weights near 0 means the (1+w) form) for the attribution computation.

```python
# Hypothesis: 12 windows <=2221 tok fit the 6GB budget; save metadata; check final-norm convention for DLA
wins2=[w_ for w_ in wins if w_['ntok']<=2250]
print(len(wins2), max(w_['ntok'] for w_ in wins2), sum(w_['ntok'] for w_ in wins2))
json.dump([{k:v for k,v in w_.items()} for w_ in wins2], open('/work/tf_wins_C.json','w'))
nw=load('model.language_model.norm.weight'); print('final norm w mean/std', float(nw.mean()), float(nw.std()))
req={"type":"teacher_forced","model":"edited","texts":[w_['text'] for w_ in wins2],
     "return":{"token_logprobs":True,"topk":10,"hidden_states":list(range(25)),"mlp_act":list(range(12,24))}}
json.dump(req, open('/work/req_tfC.json','w'))
print(len(json.dumps(req)))
```