Scale 0.75 reached 41.2% (GSM8K 53.1%, MATH 34.1%) with 51% cut, almost the same as E6's 42.6%. Over the 512 held-out problems the full curve is 11.5% at scale 0, 14.1% at 0.5, 41.2% at 0.75, 42.6% at 1.0 and 43.2% at 1.5. The transition is sharp and lies between 0.5 and 0.75. At 0.5, L23's in-subspace output is only removed and L15/L19 are flipped weakly (−0.5). At 0.75, L23 is actively reversed (−0.5) and L15/L19 flip beyond their original magnitude (−1.25). That fits the earlier finding that merely removing the copy push is compensated by other paths (the L22 MLP suppressor relaxing, L23 MLP) and only an active reversal tips the opener choice. E6 at 1.0 therefore sits on the plateau, with margin from the knee. It has fewer answers without a boxed result than 1.5, so it is the final edit, and I restore it now. I will use the last three passes for teacher-forced mechanism checks on the original model's compact loop texts, which pass 4 recorded under the original weights. These cover loop-opener probability by repetition, escape probability per period, and the change on non-loop prompt and lead-in tokens, under E6 and under the failing half strength. That should show directly why 0.5 fails and 1.0 works. First I check the size of those texts, since they must be pasted inline.

```python
# Restore final E6; log dose-response; inspect compact loop texts (sizes/structure) for inline TF mechanism passes.
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W6=torch.load(f'/work/best_E6/L{l}_o_proj.pt'); save_edit(nm, W6)
    W0=load_original(nm); print(l, 'E6 restored, max|diff to formula|', float((load(nm)-(W0-{15:3.,19:3.,23:2.}[l]*Q6@(Q6.T@W0))).abs().max()))
G[23]=readj('/work/gen_23.jsonl'); G['23H']=[r for r in G[23] if r['id'] in set(H)]; G['23H2']=[r for r in G[23] if r['id'] in set(H2)]
for src in [None,'gsm8k','math']:
    print(src, 's.75-orig', pooled([(G['18o'],G['23H']),(G['21o'],G['23H2'])], src), '| E6-s.75', pooled([(G['23H'],G['18e']),(G['23H2'],G['21e'])], src))
with open('/work/notes.md','a') as fh:
    fh.write("- PASS 22 (gen_23) scale .75 on H+H2: acc 41.2 (gsm 53.1, math 34.1), cut 50.8.\n  DOSE-RESPONSE FINAL (512 held-out): s=0 11.5 | .5 14.1 | .75 41.2 | 1.0 42.6 | 1.5 43.2. Knee between .5 and .75 (L23 removal->reversal). FINAL = E6 (s=1), restored.\n")
cp=json.load(open('/work/tf_compact.json')); print(type(cp).__name__, len(cp))
e0=cp[0]; print(list(e0.keys()) if isinstance(e0,dict) else type(e0))
lens=[len(x['text'] if isinstance(x,dict) else x) for x in cp]; print('char lens', lens, 'total', sum(lens))
print({k:(str(v)[:60] if k!='text' else '') for k,v in e0.items()} if isinstance(e0,dict) else '')
```