Run 13 gave three results.

(a) The heads whose own QK weights changed most, relative to their size, are L11 H0 (5.0%), L11 H1 (4.2%), L10 H1 (3.4%), L12 H2 (3.1%), L12 H0 (2.9%), L12 H3, L11 H3, L14 H1, L10 H0, L5 H3 and L5 H0. Among the strongest guess readers, L12 H1 changed moderately (2.4%), while L10 H2 (1.1%) and L11 H2 (1.8%) changed little. So their large attention shift in B must be largely driven by upstream changes to the residual stream: what the guess position looks like, and what the answer position's query looks like.

(b) Attending to the guess pushes the guess's logit above the correct answer's in L12 H1, L17 H0, L11 H2, L12 H3, L10 H2, L9 H2 and L9 H0. These are the heads B points at the guess, so in B they work as copy heads. L13 H3, L17 H2 and L9 H1 have the opposite sign, which suggests an indirect role.

(c) The guess token itself attends to the user's cue word ("think"/"guess"/"sure"/"believe") more in B than in A, in L10 H0 (−0.18), L6 H3 (−0.175), L8 H3 (−0.135), L6 H1 (−0.10), L7 H3 and L4 H3 (−0.08). A does so more only in L11 H0 (+0.09). So B "tags" the guess token as the user's belief early on, which makes it findable for the L9–L12 answer-position heads.

That is a two-step sycophancy circuit in B: tag the guess, then read and copy it at the answer position. A keeps reading the question. P(A syco) ≈ 0.04.

Next, I'll localize every head's attention change by query-position type in the run-2 files. The types are question tokens, cue words, guess tokens, the rest of the user turn and the model-turn/answer positions. This shows what the heavily QK-changed heads (L11 H0/H1, L10 H1, L12 H0/H2) do, and which heads differ on the key side (the guess positions) versus the query side (the answer position). For each such head I'll also list the most changed (query type → key type) attention flows.

```python
R='/work/attn/r2'; toks=[x['tokens'] for x in json.load(open(f'{R}/tokens.json'))]
G={0:[10,11],1:[10,11],2:[8],3:[8],4:[35],5:[35],6:[19],7:[14],8:[],9:[],10:[18,19],11:[26,27],12:[18,19],13:[15],14:[15],15:[18,19]}
cue_words=(' think',' guess',' sure',' believe')
types=['bos','question','cue','guess','otheruser','modelturn']
def ptype(i):
    tk=toks[i]; T=len(tk); st=max(j for j,x in enumerate(tk) if x=='<start_of_turn>')
    ty=['otheruser']*T; ty[0]='bos'
    qend=[j for j,x in enumerate(tk) if x=='?']
    # question span: from 'What/Which/How/Who' token to first '?' after it
    for j,x in enumerate(tk):
        if x in ('What','Which','How','Who',' What'):
            e=min([q for q in qend if q>j]+[T-1]); 
            for p in range(j,e+1): ty[p]='question'
            break
    for j,x in enumerate(tk):
        if x in cue_words: ty[j]='cue'
    for g in G[i]: ty[g]='guess'
    for p in range(st,T): ty[p]='modelturn'
    return ty
nt=len(types)
flow=np.zeros((18,4,nt,nt))   # sum over texts of mean |A-B| attention from query type to key type
signed=np.zeros((18,4,nt,nt))
for i in range(16):
    ty=ptype(i); ti=np.array([types.index(t) for t in ty])
    a=np.load(f'{R}/t{i}_A.npy').astype(np.float32); b=np.load(f'{R}/t{i}_B.npy').astype(np.float32)
    d=a-b
    for qt in range(nt):
        qs=np.where(ti==qt)[0]
        if len(qs)==0: continue
        for kt in range(nt):
            ks=np.where(ti==kt)[0]
            if len(ks)==0: continue
            blk=d[:,:,qs][:,:,:,ks].sum(-1)  # per query: change in mass to key type
            flow[:,:,qt,kt]+=np.abs(blk).mean(-1)/16
            signed[:,:,qt,kt]+=blk.mean(-1)/16
tot=flow.sum((2,3))
print("total |A-B| attention mass change per head (sum over query/key types):"); print(np.round(tot,2))
order=np.argsort(-tot.ravel())[:16]
for o in order:
    L,h=divmod(int(o),4)
    s=signed[L,h]
    idx=np.argsort(-np.abs(s).ravel())[:4]
    desc=[f"{types[k//nt]}->{types[k%nt]}:{s.ravel()[k]:+.3f}" for k in idx]
    print(f"L{L}H{h} tot={tot[L,h]:.2f} QKrel={['%.3f'%x for x in [0]][0]} top signed (A-B):", desc)
for (L,h) in [(11,0),(11,1),(10,1),(12,0),(12,2),(10,0),(5,0),(5,3),(6,3),(8,3)]:
    s=signed[L,h]; idx=np.argsort(-np.abs(s).ravel())[:4]
    print(f"  L{L}H{h} tot={tot[L,h]:.2f}:", [f"{types[k//nt]}->{types[k%nt]}:{s.ravel()[k]:+.3f}" for k in idx])
```