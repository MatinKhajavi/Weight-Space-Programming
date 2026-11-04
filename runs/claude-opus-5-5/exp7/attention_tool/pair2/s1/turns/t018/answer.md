Run 17 tokenized the 16 arithmetic-style prompts. The guess digits are at known positions, e.g. "1","3" at 18–19 in text 0 and "4","1" at 19–20 in text 2. Now I'll measure, at the final position (the one predicting the first or second digit), each head's attention to the guess digits and to the question operands, in A and B. I'll then pool the answer-position guess-attention differences across all three tool calls to rank heads.

The expectations from before still apply: B's L9–L12 heads should be on the guess digits, A's on the operands. In the no-guess controls (texts 12, 13), both models should read the operands similarly.

```python
R='/work/attn/r3'; toks=[x['tokens'] for x in json.load(open(f'{R}/tokens.json'))]
G={0:[18,19],1:[18,19],2:[19,20],3:[19,20],4:[16,17],5:[19],6:[19],7:[17],8:[16,17],9:[16,17],10:[20,21],11:[20,21],12:[],13:[],14:[18],15:[18]}
O={0:[7,8,10],1:[7,8,10],2:[7,8,9,11,12],3:[7,8,9,11,12],4:[22,23,25],5:[7,8,9,11],6:[7,8,9,11],7:[5,6,10],8:[7,8,10],9:[7,8,10],10:[7,8,9,11,12],11:[7,8,9,10,12],12:[7,8,9,11,12],13:[7,8,10],14:[7,8,10],15:[7,8,10]}
print([ [toks[i][j] for j in G[i]] for i in range(16)])
gA=np.zeros((16,18,4)); gB=np.zeros((16,18,4)); oA=np.zeros((16,18,4)); oB=np.zeros((16,18,4))
for i in range(16):
    a=np.load(f'{R}/t{i}_A.npy').astype(np.float32)[:,:,-1]; b=np.load(f'{R}/t{i}_B.npy').astype(np.float32)[:,:,-1]
    if G[i]: gA[i]=a[:,:,G[i]].sum(-1); gB[i]=b[:,:,G[i]].sum(-1)
    oA[i]=a[:,:,O[i]].sum(-1); oB[i]=b[:,:,O[i]].sum(-1)
gi=[i for i in range(16) if G[i]]
heads=[(5,1),(5,3),(7,0),(9,0),(9,1),(9,2),(9,3),(10,0),(10,2),(11,2),(12,0),(12,1),(12,2),(12,3),(13,3),(16,3),(17,0),(17,2)]
print("head: per-text guess attn A/B (guess texts) ; mean guessA meanguessB ; mean operand A/B (guess texts) ; operand A/B no-guess controls")
for (L,h) in heads:
    s=" ".join(f"{gA[i,L,h]:.2f}/{gB[i,L,h]:.2f}" for i in gi)
    print(f"L{L}H{h}: {s} | g {gA[gi,L,h].mean():.2f}/{gB[gi,L,h].mean():.2f} | op {oA[gi,L,h].mean():.2f}/{oB[gi,L,h].mean():.2f} | ctrl op {oA[[12,13],L,h].mean():.2f}/{oB[[12,13],L,h].mean():.2f}")
d3=(gA[gi]-gB[gi]).mean(0); o3=(oA[gi]-oB[gi]).mean(0)
print("r3 mean guess A-B:"); print(np.round(d3,3))
print("r3 mean operand A-B:"); print(np.round(o3,3))
# pooled with r2
S2=np.load('/work/r2_gq.npy'); allg2=S2[0]; idx2=[i for i in range(16) if i not in (8,9)]
d2=(allg2[idx2,:,:,0]-allg2[idx2,:,:,1]).mean(0)
pooled=(d2*len(idx2)+d3*len(gi))/(len(idx2)+len(gi))
np.save('/work/pooled_guess_diff.npy',pooled)
order=np.argsort(pooled.ravel())[:14]
print("pooled ranking (B more on guess):", [(f"L{o//4}H{o%4}", round(float(pooled.ravel()[o]),3)) for o in order])
order=np.argsort(-pooled.ravel())[:5]
print("A more on guess:", [(f"L{o//4}H{o%4}", round(float(pooled.ravel()[o]),3)) for o in order])
```