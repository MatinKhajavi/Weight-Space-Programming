V2 now matches C exactly on all 30 anchor line-start tokens, including every doubt marker and "So". It differs only on the protected tokens, and the vocabulary-wide check shows the change is confined to the concept "ready" in all its forms and languages (" Ready", "ready", "READY", " readiness", ...) plus "</think>". 99% of tokens change by less than 0.05 logit per unit activation. So the brake column carries a "ready / done thinking" component, and doubling it doubled the suppression of readiness at every new-thought line start. I will save V2 and print the S1 problem ids so I can paste them into a generation pass. Running V2 on the same 256 S1 problems and seeds as the C run (gen_13) gives a paired comparison. Fewer cut answers, more correct answers and few new wrong answers would support the hypothesis; a rise in wrong answers would mean V2 makes the model conclude prematurely.

```python
# Save V2 (C column + readiness-protecting correction) and print S1 ids for the generation request
W22v2=W22o.clone(); W22v2[:,573]=colV2
save_edit(name22, W22v2)
print('edited:', edited(), '| check:', eff_tokens(load(name22)[:,573])[' Ready'], eff_tokens(load(name22)[:,573])[' Wait'])
print(json.dumps(S1ids, separators=(',',':')))
```