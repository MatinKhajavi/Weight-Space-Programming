The forward tool needs the texts inline ("KeyError: 'texts'"), so a 1.2 MB teacher-forced corpus cannot be submitted. Dense teacher-forced passes are therefore limited to a few short windows I can paste. Generation passes with paired outcome comparisons remain the main instrument. I will test the "brake suppresses concluding" hypothesis directly with a refined brake. For that I need the set of tokens a correct thinking phase uses to conclude, so it can be protected from the brake. I will collect the line-initial words of the last few lines before "</think>" in edit-C correct answers and look up the brake column's z-score and logit effect for each of these tokens. Any concluding starter with a clearly positive score is being suppressed by the doubled brake, and goes into the protected set.

```python
# Hypothesis: the concluding lines before </think> start with a small set of tokens (Ready/Final/Construct/...) some of which the brake column suppresses (positive z)
endw=collections.Counter(); endtok=collections.Counter(); lastline=collections.Counter()
for (_,r_,s_) in allrecs:
    if not (s_['correct'] and '</think>' in s_['text']): continue
    th_=s_['text'][:s_['text'].find('</think>')]
    lines_=[l for l in th_.split('\n') if l.strip()]
    for j_,l_ in enumerate(lines_[-4:]):
        w_=re.sub(r'^[\s*\->#\d.)]+','',l_); m_=re.match(r"[A-Za-z']+", w_)
        if m_: endw[m_.group(0)]+=1
        if j_==len(lines_[-4:])-1 and m_: lastline[m_.group(0)]+=1
print('last-line first words:', lastline.most_common(15))
print('last-4-lines first words:', endw.most_common(25))
cands_conc=[' Ready',' Final',' Construct',' Output',' Format',' Done',' Everything',' Perfect',' Good',' Great',' Now',' Step',' Let',' The',' This',' Check',' Proceed',' I',' Formatting',' Wr',' Write',' Plan',' Response',' Structure','</think>']
ids_c={w:tid(w) for w in cands_conc}
D_=W22o[:,[573]]; U_=g_[:,None]*D_; sd_=float(torch.sqrt(((SigE@U_)*U_).sum()))
print({w:(i, round(float(((E_[i]-mu_E)@(g_*c573)))/sd_,1), round(float(((E_[i]-mu_E)@(g_*c573)))/rmsm,3)) for w,i in ids_c.items() if i is not None})
```