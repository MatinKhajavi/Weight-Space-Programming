Hidden states and MLP activations come back in float16. That is about 51 KB per token for all 25 residual layers, plus 7 KB per MLP layer. 14 windows of about 2,300 tokens, with all hidden states and MLP activations for layers 12-23, come to about 4.5 GB, inside the limit. Now I'll build the windows: 14 loops (7 GSM8K, 7 MATH) with periods of 30-600 characters and an onset at least 4,000 characters in. Each window is the exact prompt followed by about 1,200 thinking tokens before the onset and about 900 tokens inside the loop, cut at line boundaries.

```python
# Hypothesis: windows (prompt + ~1200 pre-onset tokens + ~900 loop tokens) capture loop entry and maintenance
enc=lambda s: tok.encode(s, add_special_tokens=False).ids if hasattr(tok,'encode') else None
try:
    print(len(enc('hello world')))
except Exception as e:
    print('enc err',e)
cands=[x for x in loopsC if 30<=x['P']<=600 and x['on']>=4000]
print('cands',len(cands), collections.Counter(x['src'] for x in cands))
rng=np.random.RandomState(0)
sel=[]
for src in ['gsm8k','math']:
    c=[x for x in cands if x['src']==src]; idx=rng.permutation(len(c))[:7]; sel+= [c[i] for i in idx]
wins=[]
for x in sel:
    t=x['text']; on=x['on']
    # char positions: take tokens around onset
    pre_ids=enc(t[:on]); post_ids=enc(t[on:])
    a=max(0,len(pre_ids)-1200)
    pre_txt=tok.decode(pre_ids[a:]) if a>0 else t[:on]
    # start at next newline for cleanliness
    j=pre_txt.find('\n'); pre_txt=pre_txt[j+1:] if (a>0 and j>=0) else pre_txt
    post_txt=tok.decode(post_ids[:900])
    head=PREFIX+PROB[x['id']]['problem']+SUFFIX
    full=head+pre_txt+post_txt
    n_head=len(enc(head)); n_pre=len(enc(head+pre_txt))
    wins.append(dict(g=x['g'],id=x['id'],src=x['src'],P=x['P'],text=full,n_head=n_head,on_tok=n_pre,ntok=len(enc(full)),unit=x['text'][-x['P']:]))
for w_ in wins: print(w_['src'][:4], w_['id'][-10:], 'P',w_['P'],'ntok',w_['ntok'],'on_tok',w_['on_tok'],'|',repr(w_['unit'][:70]))
print('total tok',sum(w_['ntok'] for w_ in wins),'max',max(w_['ntok'] for w_ in wins))
```