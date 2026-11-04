The 126 cut answers of edit C (out of 512) split into 82 exact loops and 44 long runs that hit the cap without an exact loop. Loops start late: median onset is about 55k characters, at 57% of the text. Cycle starts are varied (math lines, "Maybe", bullet items), so no single trigger token remains the way "Wait" did. The reference answer appears before onset in 56% of cut texts (GSM8K 67%, MATH 51%), so often the model had the answer and kept going. That number needs a control, though, because small integer answers appear by chance. Next I compute the same rate using other problems' answers as a control and check how much near-repetition the non-exact cut texts contain. I will also read how correct C answers end their thinking, which shows what a successful conclusion looks like. Edit C also has more finished-but-wrong answers than brake-only (28 vs 8 on S1). If the answer-before-onset rate is well above control, failure to conclude after solving is the larger remaining mechanism.

```python
# Hypothesis: (1) answer-before-onset in C cut texts is well above a random-answer control; (2) non-exact cut texts are near-loops (high line reuse); (3) see how correct C thinking ends
rng_=np.random.default_rng(0)
allrecs=[(nm_run,r_,s_) for nm_run,G_ in [('S1',gC13),('H1',gC16)] for r_ in G_ for s_ in r_['samples']]
ansl=[r_['answer'] for _,r_,_ in allrecs]
ctrl_hits=[];real_hits=[]
for (nm_run,r_,s_) in allrecs:
    if not s_['cut']: continue
    t_=s_['text']; P_,on_=tail_period(t_); pre_=t_[:on_] if on_ else t_
    real_hits.append(ans_in(pre_, r_['answer']))
    ctrl_hits.append(np.mean([ans_in(pre_, ansl[j_]) for j_ in rng_.choice(len(ansl),10) if ansl[j_]!=r_['answer']]))
print('ans before onset: real', np.mean(real_hits).round(3), 'control', np.mean(ctrl_hits).round(3))
# line reuse in last 20k chars for non-exact vs exact cut texts
def line_reuse(t_, win=20000):
    lines_=[l.strip() for l in t_.split('\n') if len(l.strip())>15]
    tailL=[l.strip() for l in t_[-win:].split('\n') if len(l.strip())>15]
    seen=collections.Counter(lines_)
    return np.mean([seen[l]>1 for l in tailL]) if tailL else np.nan
ne=[line_reuse(s_['text']) for (_,_,s_) in allrecs if s_['cut'] and tail_period(s_['text'])[0] is None]
ok_lr=[line_reuse(s_['text']) for (_,_,s_) in allrecs if s_['correct'] and len(s_['text'])>40000]
print('non-exact cut: tail line reuse median', np.nanmedian(ne).round(2), 'q25-75', np.nanpercentile(ne,[25,75]).round(2), '| long correct texts', np.nanmedian(ok_lr).round(2) if ok_lr else None, len(ok_lr))
# correct C answers: tokens and how thinking ends
okC=[(r_,s_) for (_,r_,s_) in allrecs if s_['correct']]
print('correct C tokens median', np.median([s_['tokens'] for _,s_ in okC]), 'q75', np.percentile([s_['tokens'] for _,s_ in okC],75), 'q90', np.percentile([s_['tokens'] for _,s_ in okC],90))
for r_,s_ in [okC[i] for i in [3,40,90]]:
    t_=s_['text']; k_=t_.find('</think>')
    print('...', repr(asc(t_[max(0,k_-260):k_+12])))
```