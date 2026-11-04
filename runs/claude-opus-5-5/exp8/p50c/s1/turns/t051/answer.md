Two corrections from the tool notes matter here. Separate generation passes are not reproducible, so I'll only compare models within a single "both" pass. I must also log the exact edit and problem set in notes.md before each pass, which I'll start doing now. The returned hidden_24 has RMS 0.31, the same scale as hidden_23, so it is the pre-norm residual. The final norm weight averages 3.3, so I need to know whether the norm multiplies by w or by (1+w). I'll settle that by checking which version reproduces the logit gaps between top-k tokens in the saved log-probs. Then I'll do the attention-versus-MLP split of the in-loop excess for every layer. I'm also writing the diagnosis numbers into the notes so they survive another restart.

```python
# Hypothesis: (a) find correct final-norm form via top-k logit gaps; (b) split loop excess into attn vs MLP per layer
with open('/work/notes.md','a') as f: f.write("""
- gen_3 (pass, original, pool D 512x1, 32768 tok): acc 44/512=8.6%; 470 cut at 32768 (loops), only 1 cut reached </think>; correct mean len 6.9k tok.
- Loops: exact char-periodic tails in 448/470 cut; onset median 6.2k tok (10-90%: 3.2k-11.8k); period median 119 chars; loop units mostly 'Wait, <computation line>' (368/470 contain digits).
- fwd_4 (TF, original, probe1/probe2 own looping texts 875/1544 tok, all hidden, mlp, attn): induction mass in-loop vs pre-loop: 15.7 0.89/0.47, 11.1 0.76/0.49, 23.5 0.68/0.26, 23.0 0.55/0.21, 23.7 .47/.18, 19.5 .46/.20, 19.7 .43/.18, 23.4 .43/.20
- DLA (next-token logit) in-loop excess ~+6.6 total, from layer15 (+2.5) and layer23 (+4.3), layer19 (+0.7).
""")
tkid=np.load('/work/fwd_4/topk_ids.npy',mmap_mode='r')[1,:L4[1]]; tklp=np.load('/work/fwd_4/topk_logprobs.npy',mmap_mode='r')[1,:L4[1]]
hn=H[24]/np.sqrt((H[24]**2).mean(-1,keepdims=True)+1e-6)
for nm,wf in [('w',gF),('1+w',1+gF)]:
    errs=[]
    for t in range(100,1500,50):
        z=(hn[t]*wf)@Eemb[tkid[t,:5]].T; errs.append(np.abs((z-z[0])-(tklp[t,:5]-tklp[t,0])).mean())
    print(nm,'mean abs err of logit gaps',np.mean(errs).round(3))
```