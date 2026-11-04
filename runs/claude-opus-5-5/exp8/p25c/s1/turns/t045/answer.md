The helper functions' source code is not recoverable, so I will write fresh, small helpers. The 52 finished correct answers have a median of 4.7k tokens and a maximum of 13.3k, so the full set (~260k tokens) fits in one teacher-forced pass. The measurement set I plan has three parts:
1. **Loops in their real long context:** the model's actual generated prefix up to loop onset plus about 6 periods. This is more faithful than the short "compact" excerpts and shows both entry into the loop and lock-in.
2. **Benign reference:** the 52 correct finished answers, to measure collateral damage.
3. **Continuity:** the 14 old compact texts, for comparison with the earlier pass.

Here I check how the loop table encodes positions (character offsets of loop start and period) and how the chat prompt is built, so the texts can be built correctly. Nothing here tests a hypothesis yet.

```python
# Bookkeeping: inspect loop-table fields and prompt construction so I can rebuild real-context loop texts
r0 = c2.iloc[0]
print({k: r0[k] for k in ['id','src','j','tokens','cut','correct','period','loopstart','nchar','onset_tok','period_tok']})
t0 = r0['text']; ls = int(r0['loopstart']); P0 = int(r0['period'])
print('starts:', repr(t0[:120]))
print('at loopstart:', repr(t0[ls:ls+min(P0,150)]))
print('repeat check:', t0[ls:ls+P0] == t0[ls+P0:ls+2*P0])
cp = compact[0]; print('compact p:', cp['p'], 'lead len', len(cp['lead']), 'body len', len(cp['body']))
print(repr(cp['text'][:60]), '...', repr(cp['text'][-60:]))
try:
    ps_ = prompt_str(pid2prob[r0['id']]['problem']) if isinstance(pid2prob[r0['id']], dict) else prompt_str(pid2prob[r0['id']])
    print('prompt_str ok:', repr(ps_[:80]), '...', repr(ps_[-60:]))
except Exception as e:
    print('prompt_str err', e)
print(c2.onset_tok.describe().round(0).to_dict()); print(c2.period_tok.describe().round(1).to_dict())
```