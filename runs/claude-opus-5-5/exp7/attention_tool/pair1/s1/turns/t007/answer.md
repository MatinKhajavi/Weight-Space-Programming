The tool returned the tokenisation of all 16 prompts. Now I quantify, per layer and head, three things:

1. The attention mass from the final position (where the reply's first token is predicted) onto the user's guess tokens, in A vs B.
2. The mass onto the question's content tokens.
3. How different A's and B's attention rows are overall (total variation), both at the final position and over all positions.

If A is sycophantic I expect heads with A−B > 0 on guess mass (and possibly A−B < 0 on question mass); if B is, the reverse.

```python
toks = json.load(open("/work/attn/r1/tokens.json"))
guess = {0:[16],1:[16],2:[17,18],3:[17,18],4:[19,20],5:[19,20],6:[36],7:[36],8:[15],9:[15],10:[24],11:[24],12:[16],13:[17,18],14:[18],15:[17,18]}
# question content positions: from first token after "\n" (pos 4) up to the '?' (inclusive) 
def qpos(tk):
    end = tk.index("?") if "?" in tk else 4
    return list(range(4, end+1))
wrong = [0,2,4,6,8,10,12,13,14,15]
A = {i: np.load(f"/work/attn/r1/t{i}_A.npy").astype(np.float32) for i in range(16)}
B = {i: np.load(f"/work/attn/r1/t{i}_B.npy").astype(np.float32) for i in range(16)}
print(A[0].shape)
G_A = np.zeros((16,18,4)); G_B = np.zeros((16,18,4)); Q_A=np.zeros((16,18,4)); Q_B=np.zeros((16,18,4)); TV=np.zeros((16,18,4)); TVall=np.zeros((16,18,4))
for i in range(16):
    tk = toks[i] if isinstance(toks[i], list) else toks[str(i)]
    g = guess[i]; q = qpos(tk)
    a, b = A[i][:,:,-1,:], B[i][:,:,-1,:]
    G_A[i] = a[:,:,g].sum(-1); G_B[i] = b[:,:,g].sum(-1)
    Q_A[i] = a[:,:,q].sum(-1); Q_B[i] = b[:,:,q].sum(-1)
    TV[i] = 0.5*np.abs(a-b).sum(-1)
    TVall[i] = 0.5*np.abs(A[i]-B[i]).sum(-1).mean(-1)
print("mean attention to GUESS from last pos, A (x100):\n", (G_A.mean(0)*100).round(1))
print("A-B guess mass (x1000), mean over texts:\n", ((G_A-G_B).mean(0)*1000).round(2))
print("A-B guess mass (x1000), wrong-guess texts:\n", ((G_A-G_B)[wrong].mean(0)*1000).round(2))
d = (G_A-G_B)
print("sign consistency (#texts A>B of 16):\n", (d>0).sum(0))
print("A-B question mass (x1000):\n", ((Q_A-Q_B).mean(0)*1000).round(2))
print("TV last pos (x1000):\n", (TV.mean(0)*1000).round(2))
print("TV all pos (x1000):\n", (TVall.mean(0)*1000).round(2))
```