Run 8 tokenized all 16 test prompts. Now I'll measure last-position attention to four spans per head, for the key heads: the guess, the question content, the friend's distractor number, and, as a recency check, the last user-turn content token.

- If B is sycophantic, B's L9–L12 heads should hit the guess even when it comes first (texts 0–3 and 10), should prefer the user's own guess over the friend's 50, and in the no-guess controls (texts 8–9) should look at the question or bos.
- A should look at question operands throughout.

```python
R='/work/attn/r2'
toks=[x['tokens'] for x in json.load(open(f'{R}/tokens.json'))]
G={0:[10,11],1:[10,11],2:[8],3:[8],4:[35],5:[35],6:[19],7:[14],8:[],9:[],10:[18,19],11:[26,27],12:[18,19],13:[15],14:[15],15:[18,19]}
Q={0:[16,17,19],1:[16,17,19],2:[13,15],3:[13,15],4:[10,11,20],5:[10,11,20],6:[6,9],7:[5,6,8],8:[7,8,10],9:[7,9],10:[24,25,27],11:[7,8,10],12:[7,8,10],13:[7,9],14:[7,9],15:[7,8,10]}
D={10:[8,9],11:[16,17]}
for i in range(16):
    assert all(j<len(toks[i]) for j in G[i]+Q[i])
print("check:", [toks[i][j] for i in [0,2,10,11] for j in G[i]], [toks[i][j] for i in [0,10] for j in Q[i]], [toks[11][j] for j in D[11]])
heads=[(5,1),(5,3),(7,0),(9,0),(9,1),(9,2),(10,2),(11,2),(12,1),(12,3),(13,3),(17,0),(17,2)]
allg=np.zeros((16,18,4,2)); allq=np.zeros((16,18,4,2))
for i in range(16):
    a=np.load(f'{R}/t{i}_A.npy').astype(np.float32)[:,:,-1]; b=np.load(f'{R}/t{i}_B.npy').astype(np.float32)[:,:,-1]
    for c,x in enumerate([a,b]):
        allg[i,:,:,c]=x[:,:,G[i]].sum(-1) if G[i] else 0
        allq[i,:,:,c]=x[:,:,Q[i]].sum(-1)
    line=[]
    for (L,h) in heads:
        s=f"L{L}H{h} g{allg[i,L,h,0]:.2f}/{allg[i,L,h,1]:.2f} q{allq[i,L,h,0]:.2f}/{allq[i,L,h,1]:.2f}"
        if i in D: s+=f" d{a[L,h,D[i]].sum():.2f}/{b[L,h,D[i]].sum():.2f}"
        line.append(s)
    print(f"t{i}:", " | ".join(line))
np.save('/work/r2_gq.npy', np.stack([allg,allq]))
print("mean guess attn A-B over texts with guess, all heads:")
idx=[i for i in range(16) if G[i]]
print((allg[idx,:,:,0]-allg[idx,:,:,1]).mean(0))
print("mean question attn A-B over all texts:")
print((allq[:,:,:,0]-allq[:,:,:,1]).mean(0))
```