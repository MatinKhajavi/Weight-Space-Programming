Layer-14 neuron 1443 is a broader "contrast/doubt" neuron. When negative it promotes " But"/"But" (ranks 77/78), "Wait"/" Wait" (94/120), " Actually" (210) and " maybe"/" Maybe" (727/1413), while concluding words sit at the bottom. So the pathway is L14 N1443 (general doubt) → L16 N998 (dedicated "wait" writer). However, N1443's activations are small (about -0.1 to -0.2 at line starts, near 0 inside loops). The direct push toward " But" at line starts is spread thinly over many late neurons (top contribution only 0.26). So no single further doubt neuron stands out the way N998 did, and more doubt ablations would probably give small gains. The larger remaining failure is the generic lock-in: 47% of the remaining loop cycles have no doubt word. To decide which attention heads are responsible, I compare each full-attention head's behaviour in normal reasoning (before loop onset) with its behaviour in loops. Normal behaviour is the induction score: attention to the token that followed an earlier occurrence of the current token, plus attention to the problem statement. Heads that are strongly loop-attending but have weak induction and weak prompt-reading in normal text are the safest targets, since damping them should barely affect copying numbers from the problem. If every loop-attending head is also a heavy normal-text copier or prompt reader, damping them is risky, and I will test only a mild strength.

```python
# Per-head normal-text behaviour (pre-onset): induction score, prompt-reading, vs loop congruent attention
n_prompt = [len(tok.encode(prompt_of('gsm8k_train_2901')).ids), len(tok.encode(prompt_of('gsm8k_train_3058')).ids)]
print('prompt lengths', n_prompt)
loop_cong = {}
for L in (3, 7, 11, 15, 19, 23):
    A = np.load(f'/work/fwd_4/attn_{L}.npy', mmap_mode='r'); res = []
    for j in range(2):
        ids,p,on = info[j]
        ts = np.arange(n_prompt[j]+50, on, 4)
        ind = np.zeros(8); prm = np.zeros(8); cnt = 0
        for t in ts:
            row = np.asarray(A[j][:, t, :t+1], dtype=np.float32)
            S = [s for s in range(1, t+1) if ids[s-1] == ids[t] and s-1 < t]
            prm += row[:, :n_prompt[j]].sum(1)
            if S: ind += row[:, S].sum(1); cnt += 1
        res.append((ind/cnt, prm/len(ts)))
    ind = (res[0][0]+res[1][0])/2; prm = (res[0][1]+res[1][1])/2
    print(f'L{L} induction(pre) {np.round(ind,2).tolist()} prompt-mass(pre) {np.round(prm,2).tolist()}')
```