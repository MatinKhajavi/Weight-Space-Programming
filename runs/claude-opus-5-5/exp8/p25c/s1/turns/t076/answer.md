The remaining loops fall into recognizable types:
- **Wrap-up loops:** the model has decided to stop but never closes its thinking. Examples: "Okay, I'm done. I'll write the response. The average score is 91. … Let's just re-verify the prompt again" and "Okay, I'll output the answer. Final Answer:" repeated.
- **Verification loops** with other openers: "So minimum is √3/2. Okay, I'll stop doubting myself now. Let me…" and "Let's check … ? No. Let".
- **Degenerate character loops:** e.g., "000…".

Loop periods now hold a median of 4 lines and 3 distinct openers. So the same copying of line openers is at work, now spread across "So/Let/Okay/I/The". Before choosing the next refinement, I want the share of each type among the 350 remaining loops. If wrap-up and multi-opener verification loops dominate, a natural generalization of E3 is to apply the same anti-copy treatment to the whole set of common line openers, not just "Wait". If degenerate short loops dominate, the cause is different, namely token-level copying.

```python
# Categorize remaining E3 loops (and original loops) by content of the repeated period
wrap_pat = re.compile(r"(I'll write|write the (final )?(response|answer)|Final Answer|I'm done|I am done|output the answer|ready to|stop (doubting|here|now)|Done thinking|final answer is|The answer is)", re.I)
ver_pat = re.compile(r"(check|verify|re-verify|recheck|double|Let me|Let's|Wait|confirm|Is it|correct\?)", re.I)
def categorize(text, P_):
    per = text[-P_:] if P_ > 0 else ''
    if P_ <= 6: return 'degenerate(P<=6)'
    if wrap_pat.search(per): return 'wrap-up'
    if ver_pat.search(per): return 'verification'
    return 'other'
for nm_, G_, LS_ in [('orig', G2, LS2), ('E3', G11, LS11)]:
    cats = Counter(categorize(G_[r['key']]['text'], int(r['P'])) for _, r in LS_[LS_.loop].iterrows())
    n_ = sum(cats.values()); print(nm_, {k_: f'{v_} ({v_/n_:.0%})' for k_, v_ in cats.most_common()})
# what precedes the stalled close in wrap-up loops: does the period contain explicit statement of the answer (boxed or 'answer is')?
wr = [G11[r['key']] for _, r in LS11[LS11.loop].iterrows() if categorize(G11[r['key']]['text'], int(r['P']))=='wrap-up']
print('E3 wrap-up loops:', len(wr), '| share whose problem answer appears in the period:',
      round(np.mean([ans_in(s_['text'][-int(LS11.set_index('key').loc[[k for k,v in G11.items() if v is s_][0],'P'])*2:], id2rec[[k for k,v in G11.items() if v is s_][0][0]]['answer'], s_['src']) for s_ in wr]),2))
```