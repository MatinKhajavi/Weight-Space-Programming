Line-initial words inside E6 loops form a moderate tail, and the openers already reversed are now under-represented in loops relative to good reasoning. "So" covers 5% of loop lines versus 18% of finished-correct lines, and "Wait" 0.5% versus 15%. Note that the model still uses Wait and So freely in good reasoning; only the copy-driven repetition is suppressed. The loop-specific words outside the set are "Is" (10% of loop lines vs 3% in good text), "Maybe" (6% vs 1.2%), "Then" (6.8% vs 2.4%), "No" (5.4% vs 1.9%) and "Yes" (4.4% vs 1.8%). "I" still appears in 8% of loop lines, perhaps as a tokenization variant after the bullet. Adding words one by one is whack-a-mole, so I am considering a more complete, data-defined set: the common line-initial words of the model's own thinking. That would leave loops no discourse opener to retreat to. Before building it, I need the token variants actually used after bullets and indentation, and the signatures of my basis-building helpers, so the new edit is constructed exactly like E3–E6.

```python
# Hypothesis: after bullets the opener token is a bare word (e.g. 'Is') or space variant; get helper signatures to rebuild subspace edits consistently.
print({fn: str(inspect.signature(globals()[fn])) for fn in ['basis','proj_sub','proj_out','opener_tokens','nmW','esc','opener_share','rmsn']})
for s_ in ["\n    *   Is it possible", "\n    *   Maybe the", "\n    *   I think", "\nIs it", "\n\nMaybe the", "\n    Then we", "\n    *   Yes.", " So the"]:
    ids_=tok.encode(s_).ids; print(repr(A_(s_)), ids_, [A_(tok.decode([i])) for i in ids_])
fin_top=[(w, round(finW[w]/nfin,4), round(loopW[w]/totL,4)) for w,_ in finW.most_common(30)]
print(fin_top)
```