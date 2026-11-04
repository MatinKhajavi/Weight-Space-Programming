Inside established cycles every token is near-certain (p > 0.95). The one exception is a digit in W2 at p ≈ 0.94, and even there the top choice is still the copy. So once a loop is running there is no branch point left; the decision happens during the first few repetitions. Next I'll look at loop entry line by line in W1: for each line around the onset, its text, the probability of its first content token after the bullet, and the minimum token probability in the line. This shows how fast copying locks in and at which token the model first commits to repeating.

```python
# Hypothesis: lock-in happens over the first 2-4 repeats; first content token after the bullet is the decision point
def line_table(b, start_tok, end_tok):
    seq=ids[b,:lens[b]].tolist()
    # split into lines by tokens containing newline
    lines=[]; cur=[]
    for p in range(start_tok, end_tok):
        cur.append(p)
        if '\n' in tok.decode([seq[p]]): lines.append(cur); cur=[]
    for ln in lines:
        txt=tok.decode([seq[p] for p in ln]).encode('ascii','replace').decode()
        probs=[float(np.exp(lp[b,p-1])) for p in ln]
        # first content token: first token whose decoded text has a letter or digit after the bullet
        fc=None
        for j,p in enumerate(ln):
            s=tok.decode([seq[p]])
            if re.search(r'[A-Za-z0-9$]', s): fc=j;break
        print(f"{ln[0]:5d} min{min(probs):.2f} first{probs[fc] if fc is not None else -1:.2f} | {txt.strip()[:70]!r}")
line_table(0, 560, 760)
```