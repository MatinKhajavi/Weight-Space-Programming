The main remaining copier is not a full-attention head but layer 21, a Gated DeltaNet (linear-attention) layer. At every repeated bullet its mixer adds +4.7 to +5.9 logits for the copied first word (" She") over the alternative (" If"/" So"). The MLP-21 output adds +2 to +4 on the early repeats. The growth of the margin during lock-in (0.6 → 3.9) comes from L19 and L20 mixers adding +1 to +2. L22/L23 MLPs push back by -1.5 to -2.6, presumably including the brake. DeltaNet layers do associative recall (key → value memory), so L21 likely recalls "what came after this bullet context last time". Before going deeper I'll check whether the same layer drives the other loop type. For W2 (the enumeration loop) I'll run the same attribution at its least-certain positions.

```python
# Hypothesis: L21 mixer (DeltaNet recall) is also the main copy source in the enumeration loop (W2)
seq1=ids[1,:lens[1]].tolist()
cand=[p for p in range(w_['n_head']+5, lens[1]) if np.exp(lp[1,p-1])<0.9]
print('W2 low-p positions',len(cand))
agg_mix=np.zeros(24); agg_mlp=np.zeros(24); n=0
for p in cand[:40]:
    y=seq1[p]; alt=[int(a) for a in tki[1,p-1] if int(a)!=y][0]
    mix,mlp,emb=layer_dla(1,p,y,alt); agg_mix+=mix; agg_mlp+=mlp; n+=1
    if n<=10:
        top=np.argsort(-np.abs(mix))[:3]; topm=np.argsort(-np.abs(mlp))[:2]
        print(p, repr(tstr(y)),'vs',repr(tstr(alt)),'p',round(float(np.exp(lp[1,p-1])),2),'mix',[(int(l),round(mix[l],1)) for l in top],'mlp',[(int(l),round(mlp[l],1)) for l in topm])
print('W2 mean mix by layer (>|0.5|):',[(l,round(v/n,1)) for l,v in enumerate(agg_mix) if abs(v/n)>0.5])
print('W2 mean mlp by layer (>|0.5|):',[(l,round(v/n,1)) for l,v in enumerate(agg_mlp) if abs(v/n)>0.5])
# W1 aggregate over repeated 'She' lines
A=np.array([r[3] for r in rowsD if tstr(r[1])==' She']); B=np.array([r[4] for r in rowsD if tstr(r[1])==' She'])
print('W1 She-lines n',len(A),'mean mix (>|0.5|):',[(l,round(v,1)) for l,v in enumerate(A.mean(0)) if abs(v)>0.5])
print('W1 She-lines mean mlp (>|0.5|):',[(l,round(v,1)) for l,v in enumerate(B.mean(0)) if abs(v)>0.5])
```