Zeroing an o_proj slice silences a head's output but leaves its attention pattern intact. That is why the already-ablated heads 15.7, 23.5 and 23.0 still show strong induction attention in these C windows. Among the heads that still write into the residual stream, the strongest induction attention belongs to 11.1 (0.43 in-loop, 0.64 on normal text), then 19.5, 23.4, 15.5, 23.7 and 19.7. All of them are general induction heads, as active on normal text as in loops, not loop-specific. Combined with the zero effect of removing 19.5, this means copying is now spread redundantly over many heads, and pruning more heads would damage normal computation (as edit B showed). So I will look at what precedes loop formation instead. Many C loops I read sit at very deep bullet indentation (32–80 spaces), unlike the original model's loops (indent 4). I will compare the indentation depth of the repeating lines in all C loops with the depth distribution of lines in correct C answers. I will also measure how the loop rate varies with depth. If loops concentrate at depths that correct answers rarely reach, runaway nesting is a precursor worth investigating; if depths match, indentation is a red herring.

```python
# Hypothesis: C loops occur at deep indentation that correct answers rarely reach (runaway nesting precedes loops)
def indent_of(line): return len(line)-len(line.lstrip(' '))
loop_ind=[]; 
for (_,r_,s_) in allrecs:
    if not s_['cut']: continue
    t_=s_['text']; P_,on_=tail_period(t_)
    if P_ is None: continue
    unit=t_[on_:on_+max(P_,200)]
    ls=[l for l in unit.split('\n')[1:] if l.strip()]
    if ls: loop_ind.append(np.median([indent_of(l) for l in ls]))
ok_lines=[]; ok_max=[]
for (_,r_,s_) in allrecs:
    if not s_['correct']: continue
    th_=s_['text'][:s_['text'].find('</think>')] if '</think>' in s_['text'] else s_['text']
    ii=[indent_of(l) for l in th_.split('\n') if l.strip()]
    ok_lines+=ii; ok_max.append(max(ii) if ii else 0)
ok_lines=np.array(ok_lines); loop_ind=np.array(loop_ind)
print('loop unit indentation: median', np.median(loop_ind), 'q25-75', np.percentile(loop_ind,[25,75]), 'share >=16:', (loop_ind>=16).mean().round(2), '>=24:', (loop_ind>=24).mean().round(2))
print('correct-answer lines indentation: share >=16', (ok_lines>=16).mean().round(3), '>=24', (ok_lines>=24).mean().round(3), '| max indent per correct answer median', np.median(ok_max), 'q90', np.percentile(ok_max,90))
# progression: for cut texts with loops, indentation at 10%,30%,50% of the pre-onset text
prog=[]
for (_,r_,s_) in allrecs:
    if not s_['cut']: continue
    t_=s_['text']; P_,on_=tail_period(t_)
    if P_ is None: continue
    pre_=t_[:on_]; row=[]
    for fr in (0.1,0.3,0.5,0.7,0.9):
        seg=pre_[int(fr*len(pre_)):int(fr*len(pre_))+3000]
        ii=[indent_of(l) for l in seg.split('\n')[1:] if l.strip()]
        row.append(np.median(ii) if ii else np.nan)
    prog.append(row)
print('pre-onset indentation median at 10/30/50/70/90%:', np.nanmedian(np.array(prog),0))
```