The teacher-forced pass under E4b returned full hidden states and MLP activations for the nine So-loops (lengths 150-358 tokens). Now I locate the So decision points: positions whose next token is 'So' (4272) or ' So' (1987) at a line start, after a newline, indentation or bullet. Each gets a repetition count k, the number of earlier occurrences of exactly the same line. I measure P(So) as a function of k and then do direct logit attribution with the final norm's gain and the position's RMS. The attributed quantity is the margin of So over the mean of the model's other top-20 candidates, split into embeddings, attention per layer and MLP per layer. The question is whether L15/L19, now reversed, still push So, and which components carry So's growth with k.

```python
# DLA at So decision points under E4b (fwd_13): which components push 'So' and how it grows with repetition k
F13 = '/work/fwd_13/'
ids13 = np.load(F13+'input_ids.npy'); lp13 = np.load(F13+'token_logprobs.npy')
tki13 = np.load(F13+'topk_ids.npy'); tkl13 = np.load(F13+'topk_logprobs.npy')
lens13 = [196,286,150,233,358,200,264,273,185]
SO_IDS = {4272, 1987}
think_id = tok.encode('<think>').ids[0]
dec = []   # (text i, position t predicting token t+1, k)
for i in range(9):
    L_ = lens13[i]; toks = ids13[i, :L_]
    st = int(np.where(toks == think_id)[0][-1]) + 1
    strs = [tok.decode([int(x)]) for x in toks]
    # split into lines (by tokens containing newline); record line token tuples
    line_start = st; seen_lines = collections.Counter()
    for p in range(st, L_):
        if '\n' in strs[p] or p == L_-1:
            pass
    # find So line-start positions
    for p in range(st+1, L_):
        if int(toks[p]) in SO_IDS:
            prev = strs[p-1]
            if '\n' in prev or prev.strip() in ('*','-') or (prev.strip()=='' and len(prev)>0):
                # line = tokens from p to next newline token
                q = p
                while q < L_ and '\n' not in strs[q]: q += 1
                line = tuple(int(x) for x in toks[p:q])
                k = seen_lines[line]; seen_lines[line] += 1
                dec.append((i, p-1, k, int(toks[p])))
print('decision points', len(dec), 'by k', collections.Counter(min(d[2],6) for d in dec))
# P(So) at decision points by k (token_logprobs index t gives logprob of token t+1)
pk = collections.defaultdict(list)
for i, t, k, sid in dec: pk[min(k,6)].append(np.exp(lp13[i, t]))
print('P(So) by k:', {k: (round(float(np.mean(v)),3), len(v)) for k, v in sorted(pk.items())})
print(type(Wd_all), list(Wd_all.keys())[:3] if isinstance(Wd_all, dict) else '', type(Wd_all[0]), getattr(Wd_all[0],'shape',None))
```