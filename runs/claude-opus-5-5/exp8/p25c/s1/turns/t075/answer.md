The edit holds up statistically on the diagnosis set. Against the original, accuracy is up 16.6 points with SE 2.2 (107 answers wrong→right, 22 right→wrong). Both sources gain: GSM8K +17.7 (SE 4.1), MATH +15.9 (SE 2.4). Finished answers stay 97.8% correct, now with a median of 9.5k tokens.

The remaining failures are still mostly exact loops: 93% of the cut answers, though they start later (median onset 8.0k vs 6.7k tokens). Their make-up has changed, though. "Wait" no longer appears among the loop openers. The loops now repeat lines opening with "So" (42%), "Let" (40%), "Okay" (31%), "If" (20%), or lines starting with math or symbols (65%). So the same copy-the-line-opener runaway seems to move to whichever opener is free. Before generalizing the edit to other openers, I will read a few of the remaining loops and measure opener diversity inside loop periods. A generalized edit only makes sense if these loops are again built on repeated openers, rather than on a single long repeated formula.

```python
# Inspect remaining E3 loops: what is repeated now (short excerpts), and how many distinct openers per period
rng2 = np.random.RandomState(3)
L11loop = LS11[LS11.loop].reset_index(drop=True)
for i_ in rng2.choice(len(L11loop), 6, replace=False):
    r = L11loop.iloc[i_]; s_ = G11[r['key']]; P_ = int(r['P'])
    per = s_['text'][-P_:]
    print(r['key'][0][:28], 'P', P_, 'onset', r['onset_tok'], '|', repr(per[:260]))
# opener diversity: distinct line-openers per loop period, and lines per period
divs=[]; nl=[]
for _, r in L11loop.iterrows():
    s_ = G11[r['key']]; P_ = int(r['P']); ws = loop_opener(s_['text'], P_, len(s_['text'])-3*P_)
    divs.append(len(set(ws))); nl.append(len(ws))
print('E3 loops: lines/period median', int(np.median(nl)), '| distinct openers/period median', int(np.median(divs)), '| single-line periods share', round(np.mean(np.array(nl)<=1),2))
divs2=[]; nl2=[]
for _, r in LS2[LS2.loop].iterrows():
    s_ = G2[r['key']]; P_ = int(r['P']); ws = loop_opener(s_['text'], P_, len(s_['text'])-3*P_)
    divs2.append(len(set(ws))); nl2.append(len(ws))
print('orig loops: lines/period median', int(np.median(nl2)), '| distinct openers median', int(np.median(divs2)), '| single-line share', round(np.mean(np.array(nl2)<=1),2))
```