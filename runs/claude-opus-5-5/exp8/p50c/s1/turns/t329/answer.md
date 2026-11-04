H5 confirms C on a fifth fresh held-out set:

| | Original | C |
|---|---|---|
| Accuracy | 12.9% | 59.0% |
| Cut share | 87.9% | 24.2% |

The paired difference is +46.1 ± 3.2 points, with 119 answers going wrong→right and only 1 right→wrong. All five held-out sets now agree within noise (+45 to +49).

The remaining loss is loops that start after the answer is found, inside deeply nested bullet spirals. I considered capping indentation with whitespace-token suppression. I rejected it: the model tracks the column and could rebuild deep indentation from shorter tokens, and it would be an output constraint rather than a repair of a component.

A more principled next step extends the copy-head dose-response below zero. The L23 copy heads 23.0/23.5 write directly to the logits and give 60.0% at ×0.5 and 60.7% at ×0. Setting them to ×−0.5 makes them anti-induction heads. Their attention is diffuse in normal text but sharp in exact repetition. A negative output should therefore push against the predicted loop continuation mostly inside loops, like an internal repetition penalty triggered by an n-gram match. 15.7 stays at 0, since flipping a mid-layer head has unpredictable downstream effects.

This run builds that variant, X10, logs H5, and records how deep the loop units are nested compared with successful texts.
- **Gain:** fewer cuts without more wrong answers, so accuracy above C on SH512.
- **Failure:** wrong answers rise from mis-copied numbers and expressions.

```python
# R6 hypothesis: (a) H5 confirms C (log it); (b) loop units sit at runaway nesting depth vs ok texts (record); (c) build X10 = C but L23 copy heads 23.0/23.5 at -0.5 (anti-induction): pushes against induction-predicted continuations, strongest where attention is sharp (exact loops).
import numpy as np, collections, re, json
def lead(l): return len(l)-len(l.lstrip(' '))
H5=[(r['id'],r.get('source',''),str(r.get('answer','')).strip(),s) for r in R('/work/gen_41_edited.jsonl') for s in r['samples']]
allC=rows+H5
loops=[]
for pid,src,ans,s in allC:
    if not s.get('cut'): continue
    Lp=tail_loop(s['text'])
    if Lp: 
        p,i=Lp; u=s['text'][i:i+p]; ls=[l for l in u.split('\n') if l.strip()]
        loops.append((src, max([lead(l) for l in ls]) if ls else 0))
okmax=[(src, max([lead(l) for l in s['text'].split('\n') if l.strip()] or [0])) for pid,src,ans,s in allC if s.get('correct')]
for src in ['gsm8k','math']:
    a=np.array([d for s_,d in loops if s_==src]); b=np.array([d for s_,d in okmax if s_==src])
    print(f"{src}: loop-unit indent med {np.median(a):.0f} >=24 {np.mean(a>=24):.2f} >=40 {np.mean(a>=40):.2f} (n{len(a)}) | ok-text MAX indent med {np.median(b):.0f} >=24 {np.mean(b>=24):.2f} >=40 {np.mean(b>=40):.2f} (n{len(b)})")
o23=load_original(P+'23.self_attn.o_proj.weight').clone(); o23[:,0:256]*=-0.5; o23[:,5*256:6*256]*=-0.5
for k in [P+'15.self_attn.o_proj.weight', P+'22.mlp.down_proj.weight']: save_edit(k, C_TARGET[k])
save_edit(P+'23.self_attn.o_proj.weight', o23)
chk=(load(P+'23.self_attn.o_proj.weight')[:,0:256]+0.5*load_original(P+'23.self_attn.o_proj.weight')[:,0:256]).abs().max().item()
print('edited:', [n.replace(P,'L') for n in edited()], 'X10 check', chk)
with open('/work/notes.md','a') as fh:
    fh.write("- PASS 34 = gen_41 (both, H5 fresh 128 MATH+128 GSM): orig 12.9 (cut 87.9) | C 59.0 (gsm 63.3, math 54.7, cut 24.2); diff +46.1 SE 3.2, W->R 119, R->W 1. Held-out H1-H5 all +45..+49.\n- Remaining C loops: GSM periodic 90% of cuts, unit starts with '*' bullet 70%, ref STATED (answer/total/boxed ctx) before onset 55% vs 6% shuffled control (MATH 17% vs 3%) => GSM loops mostly EXIT FAILURE after answer, inside deeply nested re-interpretation bullets ('Maybe X refers to Y', re-quoting problem). Indent-cap idea rejected (model can rebuild indent from short tokens; output-constraint not repair).\n- X10 = C but 23.0/23.5 at -0.5 (anti-induction; extends head dose 1/.5/0 below zero). Pass 35 planned (edited, SH512).\n")
```