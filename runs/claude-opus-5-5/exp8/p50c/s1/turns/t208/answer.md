The current /work/edited is exactly edit C: head 15.7 (columns 1792–2047), heads 23.0 and 23.5 (512 columns), and the single brake column 22.573. So gen_21 tested some undocumented variant that was then reverted, with no accuracy change (60.5% vs 60.2%). To avoid repeating it blindly, I will compare its behavioural fingerprint with C's: 'Wait' rate, line-initial 'Maybe' rate, lengths and per-source accuracy. A stronger brake would show fewer Wait/Hmm lines; a change near </think> would show different ending behaviour with similar early text.

```python
# Fingerprint gen_21 vs gen_13 (C): Wait/Maybe rates, ending behaviour, lengths -> infer what variant was tested
import re
def fp(g, ids):
    W=M=H=ntok=0; L=[]; cl=0; acc={'math':[], 'gsm8k':[]}
    for i in ids:
        s=g[i]; t=s['text']; nt=int(s['tokens']); ntok+=nt
        W+=len(re.findall(r'\bWait\b', t)); M+=len(re.findall(r'(?m)^\s*(?:\*\s+)?Maybe\b', t)); H+=len(re.findall(r'\bHmm\b', t))
        cl += ('</think>' in t)
        if str(s['correct'])=='True': L.append(nt)
        acc['gsm8k' if i.startswith('gsm') else 'math'].append(str(s['correct'])=='True')
    k=ntok/1000
    return dict(wait=round(W/k,2), maybe_li=round(M/k,2), hmm=round(H/k,2), closed=cl, med_len_correct=int(np.median(L)), gsm=round(np.mean(acc['gsm8k']),3), math=round(np.mean(acc['math']),3))
print('gen13 C ', fp(g13, S1)); print('gen21 ?', fp(g21, S1)); print('gen14 x1.5', fp(g14, S1))
```