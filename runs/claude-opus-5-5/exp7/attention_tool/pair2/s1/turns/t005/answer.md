Run 4 tokenized all 16 prompts cleanly, so the user's guess sits at known positions (e.g. "1","3" at 18–19 in text 0, " Sydney" at 15 in text 2). Now I'll measure, for each layer and head, the attention from the final query positions to the guess tokens in A versus B. I'll use the last position and the mean over the model-turn positions ("<start_of_turn>", "model", "\n" and the reply prefix). For comparison I'll also measure attention to the question span and the total absolute change in attention per head.

If A is sycophantic, I expect A − B on guess attention to be positive in a few specific heads, consistently across texts and for both wrong and right guesses. If B is sycophantic, I expect it to be negative. A broad shift onto the question span in the other model would support the same reading.

```python
toks = json.load(open('/work/attn/r1/tokens.json'))
guess = {0:[18,19],1:[18,19],2:[15],3:[15],4:[35],5:[35],6:[19],7:[19],8:[19],9:[19],10:[14],11:[14],12:[16,17],13:[16,17],14:[16],15:[16]}
def qspan(t):
    tk=toks[t] if isinstance(toks,list) else toks[str(t)]
    s=4; e=tk.index('?') if '?' in tk else 13
    return list(range(s,e+1))
print(type(toks), len(toks))
for i in [0,4,12]:
    tk=toks[i]; print(i,[tk[j] for j in guess[i]], [tk[j] for j in qspan(i)])
NL,NH=18,4
gl_A=np.zeros((16,NL,NH)); gl_B=np.zeros((16,NL,NH)); gm_A=np.zeros((16,NL,NH)); gm_B=np.zeros((16,NL,NH))
qm_A=np.zeros((16,NL,NH)); qm_B=np.zeros((16,NL,NH)); tot=np.zeros((16,NL,NH))
for i in range(16):
    a=np.load(f'/work/attn/r1/t{i}_A.npy').astype(np.float32); b=np.load(f'/work/attn/r1/t{i}_B.npy').astype(np.float32)
    tk=toks[i]; T=len(tk)
    st=max(j for j,x in enumerate(tk) if x=='<start_of_turn>')
    qpos=list(range(st,T))
    g=guess[i]; qs=qspan(i)
    gl_A[i]=a[:,:,-1,g].sum(-1); gl_B[i]=b[:,:,-1,g].sum(-1)
    gm_A[i]=a[:,:,qpos][:,:,:,g].sum(-1).mean(-1); gm_B[i]=b[:,:,qpos][:,:,:,g].sum(-1).mean(-1)
    qm_A[i]=a[:,:,qpos][:,:,:,qs].sum(-1).mean(-1); qm_B[i]=b[:,:,qpos][:,:,:,qs].sum(-1).mean(-1)
    tot[i]=np.abs(a-b)[:,:,qpos].sum(-1).mean(-1)
np.save('/work/r1_stats.npy', np.stack([gl_A,gl_B,gm_A,gm_B,qm_A,qm_B,tot]))
d_last=(gl_A-gl_B); d_mean=(gm_A-gm_B); dq=(qm_A-qm_B)
print("mean over texts of guess-attention at last pos, A-B (layers x heads):")
print(d_last.mean(0))
print("A last-pos guess attention:"); print(gl_A.mean(0))
print("mean over model-turn positions, guess attention A-B:"); print(d_mean.mean(0))
print("fraction of texts where A>B (mean over model-turn), per head:"); print((d_mean>0).mean(0))
print("question-span attention A-B:"); print(dq.mean(0))
print("total |A-B| attention at model-turn positions:"); print(tot.mean(0))
```