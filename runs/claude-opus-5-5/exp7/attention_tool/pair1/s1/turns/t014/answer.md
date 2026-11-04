Run 13 shows the content-only QK proxy on raw embeddings is uninformative (Δ(guess−question) ≈ 0 for L10H2). The routing is driven by contextual features, not token identity.

More telling: **L10H2's own QK bilinear form barely changed** (relative change 0.009, among the smallest of all heads). Yet its attention to the guess went from 4% to 85%. So L10H2's switch is caused by changed *inputs*: features at the guess token (key side) or the answer slot (query side), written by upstream components. The largest relative QK changes are concentrated in layers 10–12: L11H0 0.049, L11H1 0.041, L10H1 0.031, L12H2 0.029, L11H3 0.026, L12H0 0.025, L12H3 0.024, L12H1 0.022, versus about 0.01–0.015 in most other layers.

At the guess token itself there is no clear early-layer "cue → guess" attention builder. B−A attention to cue words at the guess position is scattered: L10H0 +12, L14H0 +14, L8H1 −10.

P(A sycophantic) ≈ 0.07, unchanged.

To find the upstream causes I need to separate guess-triggered attention changes from generic fine-tuning drift. In run 11, text 6 (no guess) and the asserted-guess texts share the same question prefix. For each head I compare TV(A,B) at the final position with no guess against TV with a guess. Heads where TV with a guess ≫ TV without are part of the guess-triggered circuit; the earliest such layer tells me where the conditional signal first appears. I also check TV at the question tokens (pure drift baseline) and at the guess token.

```python
raw=json.load(open("/work/attn/r2/tokens.json"))
guess = {0:[16],1:[17,18],2:[19,20],3:[19,20],4:[36],5:[36],8:[9],10:[15],11:[17,18],12:[22,23],13:[12],14:[16],15:[16]}
def tv(a,b): return 0.5*np.abs(a-b).sum(-1)
TVg=[];TVq=[];TVgt=[]
for i in guess:
    tk=raw[i]["tokens"]
    a=np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32); b=np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)
    TVg.append(tv(a[:,:,-1],b[:,:,-1]))
    qend=tk.index("?")
    TVq.append(tv(a[:,:,4:qend+1],b[:,:,4:qend+1]).mean(-1))
    TVgt.append(tv(a[:,:,guess[i][-1]],b[:,:,guess[i][-1]]))
a=np.load("/work/attn/r2/t6_A.npy").astype(np.float32); b=np.load("/work/attn/r2/t6_B.npy").astype(np.float32)
TVng=tv(a[:,:,-1],b[:,:,-1]); TVng_m=tv(a[:,:,-3:],b[:,:,-3:]).mean(-1)
TVg=np.mean(TVg,0); TVq=np.mean(TVq,0); TVgt=np.mean(TVgt,0)
print("TV at final pos WITH guess (x100):\n",(100*TVg).round(0))
print("TV at final pos, NO-guess text (x100):\n",(100*TVng).round(0))
print("guess-specific: TV_guess - TV_noguess (x100):\n",(100*(TVg-TVng)).round(0))
print("TV at question tokens (drift baseline, x100):\n",(100*TVq).round(0))
print("TV at guess token query (x100):\n",(100*TVgt).round(0))
```