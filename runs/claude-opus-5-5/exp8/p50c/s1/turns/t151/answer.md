This changes the picture for GSM8K. Under the current edit, failing GSM samples usually had the correct answer already. It appears in 61% of wrong answers and 67% of cut-off ones, against a 17-20% chance baseline, and in the cut-offs it typically appears in the first 4% of the text. The model solves the problem, then talks itself into a doubt spiral that either ends on a wrong answer or locks into a copy loop. Breaking those loops still pays: the brake turned 66 such samples wrong-to-right on S1 by keeping the model moving until it concluded. For the loop side I'll use the attention maps already in hand. At the decision positions (the token just before each line's first word), I'll measure for every full-attention head how much attention goes to the induction target, i.e. the first word of earlier lines. I'll compare in-loop lines with pre-loop lines in W1, to see which FA heads still carry line-copy induction after edit A.

```python
# Hypothesis: some remaining FA heads (e.g. 11.1, 19.x, 23.x) put much attention on the previous line's first word at decision points
Att={l:np.load(D+f'attn_{l}.npy',mmap_mode='r') for l in [3,7,11,15,19,23]}
def ind_mass(b, p, seq):
    # query position q=p-1 (predicts token at p). induction targets: positions j+1 where seq[j]==seq[q], j<q
    q=p-1; tg=[j+1 for j in range(q) if seq[j]==seq[q] and j+1<q]
    return q, tg
fcW1=first_content_positions(0,560,1169)
pre=[p for p in fcW1 if p<673]; inl=[p for p in fcW1 if p>=673]
res={}
for l in [3,7,11,15,19,23]:
    for h in range(8):
        m_pre=[];m_in=[]
        for grp,store in [(pre,m_pre),(inl,m_in)]:
            for p in grp:
                q,tg=ind_mass(0,p,seq0)
                if tg: store.append(float(Att[l][0,h,q,tg].sum()))
        res[(l,h)]=(np.mean(m_pre),np.mean(m_in))
top=sorted(res.items(), key=lambda kv:-kv[1][1])[:14]
print('npre',len(pre),'nin',len(inl))
print([(f'{l}.{h}',round(a,2),round(b,2)) for (l,h),(a,b) in top])
```