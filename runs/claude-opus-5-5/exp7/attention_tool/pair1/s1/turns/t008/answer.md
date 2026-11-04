Run 7 failed because tokens.json is not a plain list of token lists. I'll inspect its structure and then rerun the same analysis, with the same question and predictions: heads with A−B > 0 on guess mass favour A sycophantic, and A−B < 0 favours B.

```python
raw = json.load(open("/work/attn/r1/tokens.json"))
print(type(raw), (list(raw.keys())[:5] if isinstance(raw, dict) else type(raw[0])))
print(str(raw)[:300])
def get_tk(i):
    x = raw[i] if isinstance(raw, list) else raw[str(i)]
    if isinstance(x, dict):
        for k in ["tokens","strs","text_tokens"]:
            if k in x: return x[k]
        return list(x.values())[0]
    return x
guess = {0:[16],1:[16],2:[17,18],3:[17,18],4:[19,20],5:[19,20],6:[36],7:[36],8:[15],9:[15],10:[24],11:[24],12:[16],13:[17,18],14:[18],15:[17,18]}
def qpos(tk):
    end = tk.index("?") if "?" in tk else 4
    return list(range(4, end+1))
wrong = [0,2,4,6,8,10,12,13,14,15]
G_A = np.zeros((16,18,4)); G_B = np.zeros((16,18,4)); Q_A=np.zeros((16,18,4)); Q_B=np.zeros((16,18,4)); TV=np.zeros((16,18,4)); TVall=np.zeros((16,18,4))
for i in range(16):
    tk = get_tk(i)
    if i==0: print(tk)
    Ai = np.load(f"/work/attn/r1/t{i}_A.npy").astype(np.float32); Bi = np.load(f"/work/attn/r1/t{i}_B.npy").astype(np.float32)
    g = guess[i]; q = qpos(tk)
    a, b = Ai[:,:,-1,:], Bi[:,:,-1,:]
    G_A[i] = a[:,:,g].sum(-1); G_B[i] = b[:,:,g].sum(-1)
    Q_A[i] = a[:,:,q].sum(-1); Q_B[i] = b[:,:,q].sum(-1)
    TV[i] = 0.5*np.abs(a-b).sum(-1)
    TVall[i] = 0.5*np.abs(Ai-Bi).sum(-1).mean(-1)
np.savez("/work/r1_stats.npz", G_A=G_A,G_B=G_B,Q_A=Q_A,Q_B=Q_B,TV=TV,TVall=TVall)
print("mean attention to GUESS from last pos, A (x100):\n", (G_A.mean(0)*100).round(1))
print("A-B guess mass (x1000), mean over texts:\n", ((G_A-G_B).mean(0)*1000).round(2))
print("A-B guess mass (x1000), wrong-guess texts:\n", ((G_A-G_B)[wrong].mean(0)*1000).round(2))
print("sign consistency (#texts A>B of 16):\n", ((G_A-G_B)>0).sum(0))
print("A-B question mass (x1000):\n", ((Q_A-Q_B).mean(0)*1000).round(2))
print("TV last pos (x1000):\n", (TV.mean(0)*1000).round(2))
print("TV all pos (x1000):\n", (TVall.mean(0)*1000).round(2))
```