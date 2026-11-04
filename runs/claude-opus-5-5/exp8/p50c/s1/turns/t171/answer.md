Most of the extra wrong answers under A+brake come from problems that were cut under brake-only: 21 cut→wrong, 49 cut→correct and 62 still cut, against only 3 correct→wrong. So the head ablation mainly lets hard problems finish and does little damage to reasoning that already worked. A weaker or more selective ablation is therefore unlikely to recover much. The largest remaining failure is still running out of tokens: long reasoning with late loops, where about a third of cases had solved the problem and then kept doubting. With "Wait" suppressed, doubt now opens lines with other words ("Maybe", "Is it possible", "Double check", "But"). Next I count line-initial words per 1,000 tokens in C texts, split into correct, cut and wrong, and compare with the original model. If one doubt marker is strongly enriched in cut texts, that marker is the next candidate for a brake-like mechanism, analogous to neuron 22.573 for "Wait". If doubt is spread thinly over many words, a different lever is needed.

```python
# Hypothesis: under C, re-verification moved to other line-initial doubt markers (Maybe/But/Is/Double/Actually...) that are enriched in cut texts vs correct texts
def line_initial_words(t_):
    out_=collections.Counter()
    for l_ in t_.split('\n'):
        w_=re.sub(r'^[\s*\->#\d.)]+','',l_)          # strip indentation, bullets, numbering
        m_=re.match(r"[A-Za-z']+", w_)
        if m_: out_[m_.group(0)]+=1
    return out_
def rate_table(samples_):
    tot_=collections.Counter(); ntok_=0
    for s_ in samples_:
        tot_.update(line_initial_words(s_['text'])); ntok_+=s_['tokens']
    return {w:1000*c/ntok_ for w,c in tot_.items()}, ntok_
grpC={'C_ok':[s_ for (_,r_,s_) in allrecs if s_['correct']], 'C_cut':[s_ for (_,r_,s_) in allrecs if s_['cut']],
      'C_wrong':[s_ for (_,r_,s_) in allrecs if not s_['correct'] and not s_['cut']],
      'orig':[r_['samples'][0] for r_ in gO16]}
rt={k:rate_table(v)[0] for k,v in grpC.items()}
words_=['Wait','Maybe','But','Is','Double','Actually','Alternatively','Hmm','Let','So','If','This','However','Check','Hold','Or','What','Does','Did','The','Therefore','Thus','Final','Ready','Okay','Hmmm','Hm','Re','Verify','Self','Recheck','Correct','Yes','No']
tabw=pd.DataFrame({k:[round(rt[k].get(w,0),2) for w in words_] for k in rt}, index=words_)
print(tabw.T.to_string())
# top line-initial words overall in C_cut relative to C_ok
ratio_=sorted([(w, rt['C_cut'][w], rt['C_ok'].get(w,0)) for w in rt['C_cut'] if rt['C_cut'][w]>0.3], key=lambda x:-(x[1]+0.05)/(x[2]+0.05))[:20]
print([(w, round(a,2), round(b,2)) for w,a,b in ratio_])
```