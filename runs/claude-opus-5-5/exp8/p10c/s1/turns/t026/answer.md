```python
# Hypothesis: pick ~32 loop samples with moderate onset for a dense teacher-forced study; compute token lengths.
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
prob_by_id = {p['id']: p for p in gsm + mth}
def prompt_text(pid):
    return "<|im_start|>user\n" + (prob_by_id[pid]['problem'] + SUF).strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
cand = dl[(dl.onset >= 1200) & (dl.onset <= 3600)].copy()
print(len(cand), cand.src.value_counts().to_dict())
cand['cut_len'] = [min(1400, max(12*p, 700)) for p in cand.p]
cand['tot'] = cand.onset + cand.cut_len
rngs = np.random.RandomState(0)
sel = cand.sample(n=min(32, len(cand)), random_state=0).sort_values('tot')
tf_texts = []; tf_meta = []
for r in sel.itertuples():
    s_ = samples[r.i]; ids = toks(s_['text'])
    gen_part = tok.decode(ids[:r.onset + r.cut_len])
    txt = prompt_text(s_['id']) + gen_part
    tf_texts.append(txt)
    tf_meta.append(dict(i=int(r.i), id=s_['id'], p=int(r.p), onset=int(r.onset), cut_len=int(r.cut_len), n_prompt=len(toks(prompt_text(s_['id'])))))
lens = [len(toks(t)) for t in tf_texts]
print('n texts', len(tf_texts), 'max len', max(lens), 'sum', sum(lens), 'slots', len(lens)*max(lens))
print(sorted(lens)[:5], sorted(lens)[-5:])
```