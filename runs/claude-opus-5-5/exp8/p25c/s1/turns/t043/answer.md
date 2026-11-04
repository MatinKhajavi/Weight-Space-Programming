The definitions are now clear. In `AT`, `ind` is the attention mass on induction targets (the token right after each earlier copy of the current token), `same` is the mass on earlier copies of the same token, and `k` is the repetition index. So table 1 lists induction heads: 23.5 (0.59), 15.7 (0.70), 11.1 (0.57), 23.4, 23.7, 23.0. Table 2 lists "same-token" heads, and 23.6 (0.74) is one of them, not a sink. A same-token head in the final layer can also copy: at layer 22, the earlier copy of the current token already encodes the prediction of the next token. Next I check how these attention masses change with the repetition index k. Heads whose copy attention rises with k are candidate drivers of the self-reinforcement. Heads that are flat are only generally copy-oriented.

```python
# Hypothesis: the self-reinforcing heads are those whose induction/same-token attention rises with repetition index k
pv_ind = AT.groupby(['l','h','k'])['ind'].mean().unstack('k')
pv_same = AT.groupby(['l','h','k'])['same'].mean().unstack('k')
print('rows per k:', AT[(AT.l==23)&(AT.h==5)].groupby('k').size().to_dict())
sel = [(11,1),(11,3),(15,5),(15,7),(19,0),(19,2),(19,5),(19,6),(23,0),(23,1),(23,3),(23,4),(23,5),(23,7),(23,2),(23,6),(3,7),(7,6),(7,7)]
print('IND by k'); print(pv_ind.loc[sel].round(2).to_string())
print('SAME by k'); print(pv_same.loc[sel].round(2).to_string())
```