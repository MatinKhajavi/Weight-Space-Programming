The corpus is 70 conclusion windows from edit-C correct answers (172k tokens) plus 62 post-solution windows from cut answers that had already written the reference answer (180k tokens). Padded output with layer-22 MLP activations and the final hidden state comes to about 4 GB, under the cap. I will save the full request to /work/req_tf_conc.json, the texts alone to a second file and the window metadata separately. I will also update the notes with this session's findings: the 19.5 revert, the failure breakdown, the shift of doubt to "Maybe", and the brake's off-target suppression of " Ready"/"</think>". The next reply submits the pass with "model": "edited", now exactly C again, which is the model that produced these texts.

```python
# Save TF request for the conclusion corpus + metadata; update notes with this session's findings
corp=selA+selB
req_conc={"type":"teacher_forced","model":"edited","texts":[c['text'] for c in corp],
          "return":{"token_logprobs":True,"topk":10,"hidden_states":[24],"mlp_act":[22]}}
json.dump(req_conc, open('/work/req_tf_conc.json','w'))
json.dump([c['text'] for c in corp], open('/work/tf_conc_texts.json','w'))
json.dump([{k:v for k,v in c.items() if k!='text'} for c in corp], open('/work/tf_conc_meta.json','w'))
print(os.path.getsize('/work/req_tf_conc.json')//1000, 'kB;', len(corp), 'texts')
with open('/work/notes.md','a') as f:
    f.write("""
## Session after restart (runs 141+)
- fwd_18 (C) / fwd_19 (C + head 19.5 zeroed) on 2 C loop windows: 19.5 ablation changes in-loop logprob only -0.02/-0.03 nats (in-loop copying under C is ~deterministic, mean lp -0.004/-0.010) -> 19.5 NOT a loop driver; reverted. /work/edited = C (15.7, 23.5, 23.0 zeroed; 22.573 x2).
- C failures (S1+H1, 512 answers): 126 cut (82 exact loops, onset median 55k chars, 57% into text), 75 wrong. Ans before onset 56% vs control 31% (~1/3 of cuts had solved the problem). Brake-only -> A+brake cross-tab (S1): extra wrongs come from cut->wrong (21), only 3 ok->wrong: head ablation not damaging finished reasoning.
- Line-initial words per 1k tok: 'Maybe' C_cut 3.20 vs C_ok 1.06 (orig 0.53): doubt impulse re-expressed as 'Maybe' lines after Wait braked.
- 22.573 col z: Wait 14-15, Hmm 4.6, Ready 4.0, Actually 3.1, </think> 2.2, Maybe 0.4. Final residual RMS ~0.28. Per unit act: Wait 0.83 logit, Ready 0.23, </think> 0.125. Doubling at act -4 adds: Wait -3.3, Ready -0.9, </think> -0.5, Hmm -1.05 => off-target suppression of concluding tokens.
- 22.573 act at line starts (C windows): Let -5.9, Is -5.5, No -3.6, So -3.1, If -2.8, But -2.3, '$' -1.1 => fires at 'new thought' line starts generally.
- Planned TF pass: /work/req_tf_conc.json (70 C-correct endings + 62 cut post-solution windows, 352k tok; logprobs, topk10, hidden 24, mlp 22), model edited (=C).
""")
```