Line-start detection found no positions at all. In these windows every line opens with a newline, an indentation token, a bullet token and more spaces before the first word, so the first word lies more than two tokens after the newline, and my filter then rejected it. I will redefine a line start properly: after each newline, skip whitespace and bullet tokens, and take the position predicting the first content token. Then I rerun the same comparison of per-head contributions at in-loop line starts, pre-loop line starts and within-line positions. The hypothesis is unchanged: a head that drives the line-start repeat but matters little within lines would be a surgical target.

```python
# Fix line-start detection (skip indentation/bullet tokens after newline) and redo per-head DLA at line starts
def line_start_positions(s_):
    pos=set(); L_=len(s_); i=0
    while i<L_:
        if '\n' in tok.decode([s_[i]]):
            j=i+1
            while j<L_ and tok.decode([s_[j]]).strip() in ('','*','-','**'): j+=1
            if j<L_: pos.add(j-1)        # position predicting the first content token
            i=j
        else: i+=1
    return pos
ls_tab={}
for b_ in range(2):
    L_=m18['lengths'][b_]; s_=ids_all[b_]; inl,pre=regs[b_]; LS=line_start_positions(s_)
    inl_ls=[t for t in inl if t in LS]; pre_ls=[t for t in pre if t in LS]; pre_in=[t for t in pre if t not in LS]
    for l_ in [15,19,23]:
        o_=head_dla(l_,b_)
        for h_ in range(8):
            v=o_[h_]; ls_tab.setdefault(f'{l_}.{h_}',[]).append((v[inl_ls].mean(), v[pre_ls].mean(), v[pre_in].mean(), len(inl_ls), len(pre_ls)))
lt=pd.DataFrame({k:{'loop_ls':np.mean([x[0] for x in v]),'pre_ls':np.mean([x[1] for x in v]),'pre_within':np.mean([x[2] for x in v])} for k,v in ls_tab.items()}).T
lt['ls_excess']=lt.loop_ls-lt.pre_ls
print('n line starts in-loop/pre per window:', [(x[3],x[4]) for x in ls_tab['15.0']])
print(lt.sort_values('loop_ls',ascending=False).head(10).round(3).to_string())
```