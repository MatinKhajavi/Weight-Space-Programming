The attention layout is now clear. q_proj stacks, for each of the 8 heads, a 256-row query block followed by a 256-row output-gate block. Each head's output occupies 256 columns of o_proj, and only 25% of each head's dimensions are rotary. Next I'll build a reusable teacher-forced test corpus in the exact evaluation format with two parts. Part (a): about 70 of the model's own looping texts with early onsets, each from the start of the answer through about 6 loop cycles. Part (b): the 44 correct answers, truncated to 4,500 tokens, as a sample of normal reasoning. On these I can measure, per configuration, log-probs on pre-loop reasoning, on the first repeat (loop entry) and on later cycles, plus on correct reasoning. This block only builds and checks the corpus.

```python
# Hypothesis: none (setup) - build TF corpus: early-onset loop texts (to 6 cycles) + correct texts, with region boundaries
PREFIX='<|im_start|>user\n'; SUFFIX='\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
probs={r['id']:r for r in rows}
cand=[(r,s,x) for (r,s),x in zip(cut_s,LP) if x[1] is not None and x[3]<=4300 and 20<=x[2]<=700]
rng=np.random.default_rng(0); sel_loop=[cand[i] for i in sorted(rng.choice(len(cand),size=min(72,len(cand)),replace=False))]
corpus=[]; meta_c=[]
for r,s,x in sel_loop:
    on,P=x[1],x[2]; head=PREFIX+r['problem'] if 'problem' in r else None
    prob=[json.loads(l) for l in []]
    corpus.append((r['id'],'loop',s['text'][:on+6*P],on,P))
for r,s in ok:
    enc=tok.encode(s['text']); cut_c=enc.offsets[min(4500,len(enc.ids))-1][1]
    corpus.append((r['id'],'good',s['text'][:cut_c],None,None))
print(len(sel_loop),'loop texts;', len(ok),'good texts; rows have problem?', 'problem' in rows[0])
```