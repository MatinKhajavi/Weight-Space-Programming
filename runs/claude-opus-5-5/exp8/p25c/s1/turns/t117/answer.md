The opener statistics separate two groups. Several openers the model uses normally are not over-represented in loops: ' So' (22% of loop line starts vs 24% in finished thinking), ' Is', ' Then', ' Maybe', ' What'. ' Wait' is still 15% of line starts in E5's finished reasoning, so self-correction has not been removed. Strongly loop-specific are the bare line-initial forms 'This' (31× over-represented), 'I' (25×), 'The' (10×), and also 'But' (4×) and ' I' (5×). Before extending the subspace to these, I want direct evidence that the same mechanism carries them: attention in L15/L19/L23 copying the opener more strongly with each repetition. fwd_15 already contains loops opened by ' I' ("I'll output 1825.", "I'll make sure to ...") and 'But' / ' But' ("But I need to check ...", "But usually ..."). I run the same per-component DLA at those line starts, grouped by repetition count.

```python
# DLA at line starts whose next token is a non-set opener (' I','I','But',' But','This',' This') in fwd_15 loops: is attention copy the grower?
NEW_OP = {353: ' I', 40: 'I', 3850: 'But', 1921: ' But', 1919: 'This', 1061: ' This'}
recs_n = []
for i in range(6):
    L_ = lens15[i]; toks = ids15[i, :L_]
    st = int(np.where(toks == think_id)[0][-1]) + 1
    strs = [tok.decode([int(x)]) for x in toks]
    seen_l = collections.Counter()
    for p in range(st+1, L_):
        tid = int(toks[p])
        if tid in NEW_OP:
            prev = strs[p-1]
            if '\n' in prev or prev.strip() in ('*','-') or (prev.strip()=='' and len(prev)>0):
                q = p
                while q < L_ and '\n' not in strs[q]: q += 1
                lt = ''.join(strs[p:q]); k = seen_l[lt]; seen_l[lt] += 1
                t = p-1
                x24 = np.asarray(H15[24][i, t]); rms24 = np.sqrt((x24**2).mean()+1e-6)
                alt = [(int(a), float(np.exp(lv))) for a, lv in zip(tki15[i, t], tkl15[i, t]) if int(a) != tid]
                ws = sum(w for _, w in alt); ualt = sum(w*WUn_[a] for a, w in alt)/ws
                u = gFn_*(WUn_[tid]-ualt)/rms24
                c = {'i': i, 'tid': NEW_OP[tid], 'k': min(k,4), 'p': float(np.exp(lp15[i, t]))}
                for l in range(24):
                    dl = np.asarray(H15[l+1][i, t]) - np.asarray(H15[l][i, t]); mo = Wd_all[l] @ np.asarray(M15[l][i, t])
                    c[f'a{l}'] = float((dl-mo)@u); c[f'm{l}'] = float(mo@u)
                c['total'] = float(x24@u)
                recs_n.append(c)
DN = pd.DataFrame(recs_n)
print(DN.groupby(['tid']).size().to_dict(), ' by k', DN.groupby('k').size().to_dict())
cols_n = [c for c in DN.columns if c[0] in 'am' and c[1:].isdigit()]
print('P(opener) by k:', DN.groupby('k')['p'].mean().round(2).to_dict(), ' total margin by k:', DN.groupby('k')['total'].mean().round(2).to_dict())
gk = DN.groupby('k')[['a15','a19','a23','a21','a22','m22','m23']].mean().round(2)
print(gk.to_string())
grow_n = (DN[DN.k>=2][cols_n].mean() - DN[DN.k<=0][cols_n].mean()).sort_values()
print('growth k>=2 minus k=0, top +:', grow_n[::-1][:8].round(2).to_dict())
print('mean contributions top +:', DN[cols_n].mean().sort_values()[::-1][:8].round(2).to_dict())
```