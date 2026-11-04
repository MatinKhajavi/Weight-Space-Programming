The low-index dominance is a genuine property of this model: neurons with small indices have 4 to 10 times larger activations in every layer, so the neurons appear sorted by magnitude. That is not an export artifact (22.573 still reaches -5.97 as before). So the MLP result stands, and MLPs contribute little loop-specific push. Inside a loop, almost all the copying is deterministic and carried by general-purpose heads. The only real choice point in a line stutter is the start of each line, where the model either begins something new or reproduces the previous line. I will narrow the head decomposition to those line-start positions: per head, the contribution to the line's first token at in-loop line starts, at pre-loop line starts, and at ordinary within-line positions. A head that pushes the repeat strongly at line starts but matters little within lines would be a specific line-start copier, removable at little cost. If the line-start push comes from the same heads that dominate within-line prediction, no surgical head edit for stutters exists.

```python
# Hypothesis: a specific head drives the line-start copy decision in stutters (large DLA at in-loop line starts) while contributing little within lines
ls_tab={}
for b_ in range(2):
    L_=m18['lengths'][b_]; s_=ids_all[b_]; inl,pre=regs[b_]
    is_ls=np.array([('\n' in tok.decode(s_[max(0,t-2):t+1])) and tok.decode([s_[t+1]]).strip() not in ('','*','-') if t+1<L_ else False for t in range(L_)])
    for l_ in [15,19,23]:
        o_=head_dla(l_,b_)
        for h_ in range(8):
            v=o_[h_]
            inl_ls=[t for t in inl if is_ls[t]]; pre_ls=[t for t in pre if is_ls[t]]; pre_in=[t for t in pre if not is_ls[t]]
            ls_tab.setdefault(f'{l_}.{h_}',[]).append((v[inl_ls].mean() if inl_ls else np.nan, v[pre_ls].mean() if pre_ls else np.nan, v[pre_in].mean(), len(inl_ls), len(pre_ls)))
lt=pd.DataFrame({k:{'loop_ls':np.nanmean([x[0] for x in v]),'pre_ls':np.nanmean([x[1] for x in v]),'pre_within':np.nanmean([x[2] for x in v])} for k,v in ls_tab.items()}).T
print('n line starts in-loop/pre per window:', [(x[3],x[4]) for x in ls_tab['15.0']])
print(lt.sort_values('loop_ls',ascending=False).head(10).round(3).to_string())
```