Each compact item stores the lead, the loop body and the period `p`. To label every token by region (prompt/lead vs loop period k), I'll tokenize the lead and body for texts 4 and 13. Then I'll check that they line up with the 392/416 ids stored in fwd_5 and convert `p` to a token period.

```python
# Map compact-text structure (lead/body/period) onto fwd_5 token positions for texts 4 and 13
for b,j in enumerate([4,13]):
    it=tc[j]; n_all=len(tok.encode(it['text']).ids); n_body=len(tok.encode(it['body']).ids); n_lead=len(tok.encode(it['lead']).ids)
    print(j, it['id'], 'p', it['p'], 'chars text/lead/body', len(it['text']), len(it['lead']), len(it['body']), 'toks all/lead/body', n_all, n_lead, n_body, 'fwd len', lens5[b],
          'match', bool((np.array(tok.encode(it['text']).ids)[:lens5[b]]==ids5[b,:lens5[b]]).all()) if n_all>=lens5[b] else 'short')
    print('   body start:', asc(it['body'][:120]).replace('\n','|'))
```