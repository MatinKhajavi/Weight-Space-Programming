The teacher-forced pass worked: 2 windows of about 1,170 tokens with all hidden states, MLP activations and attention maps of the current edited model. I'll find where the loop could branch. For each window I'll segment the in-loop tokens into cycles and print the mean probability of the realized token at each position within the cycle, plus the main competitor at the low-probability positions. Positions where the copy is not near-certain are the decision points to attribute next. If almost every position is near p=1 except the line start, the loop is decided at the bullet-start token as with "Wait" before. Otherwise the decision sits mid-line.

```python
# Hypothesis: loop cycles have 1-2 branch points (line start / content start) where p(copy)<0.9
D='/work/fwd_18/'
ids=np.load(D+'input_ids.npy'); lp=np.load(D+'token_logprobs.npy'); tkl=np.load(D+'topk_logprobs.npy'); tki=np.load(D+'topk_ids.npy')
lens=[1169,1174]
def tstr(i): 
    s=tok.decode([int(i)]); return s.encode('ascii','replace').decode()
for b,(w_,unitP) in enumerate([(W1,None),(W2,None)]):
    L=lens[b]; on=w_['on_tok']
    seq=ids[b,:L].tolist()
    # find cycle length in tokens: smallest T such that seq[i]==seq[i-T] for most of the tail
    tail=seq[-300:]
    T=None
    for T_ in range(2,200):
        m=np.mean([tail[i]==tail[i-T_] for i in range(T_,len(tail))])
        if m>0.97: T=T_;break
    print('win',b,'L',L,'on_tok',on,'cycle T',T)
    # positions in last 6 cycles; logprob of token at position i is lp[b,i-1]
    st=L-6*T
    for k in range(T):
        pos=[st+c*T+k for c in range(6) if st+c*T+k < L]
        pr=np.exp([lp[b,p-1] for p in pos]).mean()
        if pr<0.95:
            p=pos[-1]
            alts=[(tstr(tki[b,p-1,j]), round(float(np.exp(tkl[b,p-1,j])),3)) for j in range(4)]
            print('  k',k,'tok',repr(tstr(seq[p])),'mean p',round(pr,3),'top@last',alts)
```