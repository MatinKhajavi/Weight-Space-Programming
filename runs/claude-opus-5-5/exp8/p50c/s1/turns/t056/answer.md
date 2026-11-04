The corpus has 116 texts and 433,854 tokens, within the limit. Loop onsets sit at 2.6k-4.1k tokens and cycles have a median of 38 tokens. There is a practical problem: a teacher-forced request takes literal texts, and pasting about 1.5 million characters into a reply isn't feasible. I'll try a request that points to a file of texts. If the tool doesn't support that, the request fails and, per the rules, doesn't count. Then I'll redesign around short literal texts or generation passes. Notes first: this would be the baseline (original weights, no edits) on /work/tf_corpus1.json.

```python
# Hypothesis: none (setup) - save texts list for a file-referenced TF request; log plan in notes
json.dump([t for t,_ in TF],open('/work/tf_corpus1_texts.json','w'))
with open('/work/notes.md','a') as f: f.write("""
- TF corpus1 (/work/tf_corpus1.json): 72 own loop texts from gen_3 (onset<=4.3k tok, through 6 cycles; recs have a0, on, cyc token idx) + 44 own correct texts (<=4500 tok). 433,854 tokens.
- Planned pass: TF baseline, model original, no edits, corpus1, token_logprobs+topk10.
""")
print(os.path.getsize('/work/tf_corpus1_texts.json'))
```