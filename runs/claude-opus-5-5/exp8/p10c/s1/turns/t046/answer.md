```python
# Hypothesis: some full-attn heads are induction heads (attend to t-p+1, i.e. the token after the previous occurrence) in the loop
fa = sorted([f for f in fl if f.startswith('attn_')], key=lambda s:int(s[5:-4]))
for f in fa:
    A = np.load('/work/fwd_4/'+f, mmap_mode='r')
    line = [f, str(A.shape), str(A.dtype)]
    for j in range(2):
        ids,p,on = info[j]
        ts = np.arange(on+p, 1949)
        a = np.asarray(A[j][:, ts, :], dtype=np.float32)          # H x T x S
        ind = a[:, np.arange(len(ts)), ts-p+1].mean(1)
        cong = np.zeros(a.shape[0])
        for k,t in enumerate(ts):
            prev = np.arange(t+1-p, -1, -p); cong += a[:, k, prev].sum(1)
        cong /= len(ts); sink = a[:, :, 0].mean(1); selfm = a[:, np.arange(len(ts)), ts].mean(1)
        line.append(f"T{j} ind {np.round(ind,2).tolist()} cong {np.round(cong,2).tolist()} sink {np.round(sink,2).tolist()}")
    print('\n  '.join(line))
```