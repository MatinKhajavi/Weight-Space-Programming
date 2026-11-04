The teacher-forced pass succeeded and the returned lengths (310, 297, ... 416) exactly match my local tokenization, so the transcription was exact and positions align. The experimenters also note that attention probabilities now work, that /work/eval_prompt_format.txt gives the exact evaluation prompt, and that I should record the current edit and problem set in notes before every pass. In this step I will check the evaluation prompt format against my rendered template and update /work/notes.md with the findings so far (90% loops, all finished answers correct, loops start ~6.7k tokens). Then I'll analyze the per-token log-probabilities by repetition index within each loop text: if copy log-probabilities climb close to 0 by the 2nd-3rd repetition, the loops are self-reinforcing copying, which is my working hypothesis.

```python
# Check eval prompt format; update notes; load TF results and check alignment
print(open('/work/eval_prompt_format.txt').read()[:1500])
with open('/work/notes.md','a') as f:
    f.write("""
## Findings (original model)
- gen_2 (pass 2): D set, n=2, 32768 max: acc 10.4% (gsm 14.6%, math 7.8%); 90% cut at max tokens; ALL 52 finished answers correct.
- 443/460 cut answers end in an exact verbatim loop (median period ~120 chars / ~30 tokens); loop onset median ~6.7k tokens (min ~1k).
- Looping texts rarely reach wrap-up phrases; failure = entering self-sustaining repetition loop, not failure to emit </think>.
- Weight scan: no obvious outlier heads in full-attn layers; DeltaNet A_log/dt_bias look normal.
- fwd_4 (pass 3, TF original): 14 compact loop texts (/work/tf_compact.json): prompt + 120 tok lead + 6 periods. logprobs, top20, hidden 0-24, mlp 0-23.
- Current edit: NONE.
""")
F4 = '/work/fwd_4/'
meta4 = json.load(open(F4+'meta.json')); print(list(meta4.keys())[:10])
ids4 = np.load(F4+'input_ids.npy'); lp4 = np.load(F4+'token_logprobs.npy')
print(ids4.shape, lp4.shape, ids4.dtype, lp4.dtype)
ok = all(list(ids4[i,:len(tok_ids(compact[i]['text']))]) == tok_ids(compact[i]['text']) for i in range(14))
print('alignment ok:', ok)
```