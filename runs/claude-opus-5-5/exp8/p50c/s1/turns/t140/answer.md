As expected, there is no file option, so teacher-forced texts must be pasted inline. Per-head attention maps are only available for 2 texts anyway, so I'll do the localization with two inline windows from the current edit's loops. W1 is a GSM8K bullet-line loop ("*   She puts *all* 300 strawberries into buckets.", period 54). W2 is a MATH enumeration that first repeats one item nine times and then cycles over four items. Each window is the exact prompt plus about 600 tokens before the onset and about 500 loop tokens, with attention maps for all six full-attention layers and all hidden states and MLP activations. I'll emit the texts JSON-escaped (ensure_ascii), so I can paste them exactly. First I'll build the windows and check their sizes.

```python
# Hypothesis: two ~1.1k-token inline windows (bullet loop GSM + enumeration loop MATH) suffice for head-level localization
def mkwin(pid, npre=600, npost=500):
    x=[l for l in loopsC if l['id']==pid][0]; t=x['text']; on=x['on']
    pre_ids=enc(t[:on]); post_ids=enc(t[on:])
    pre_txt=tok.decode(pre_ids[-npre:]); j=pre_txt.find('\n'); pre_txt=pre_txt[j+1:]
    post_txt=tok.decode(post_ids[:npost])
    head=PREFIX+PROB[pid]['problem']+SUFFIX
    full=head+pre_txt+post_txt
    return dict(id=pid,P=x['P'],text=full,n_head=len(enc(head)),on_tok=len(enc(head+pre_txt)),ntok=len(enc(full)))
W1=mkwin('gsm8k_train_4031'); W2=mkwin('math_train_number_theory_780')
for w_ in (W1,W2):
    js=json.dumps(w_['text']); print(w_['id'],'ntok',w_['ntok'],'on_tok',w_['on_tok'],'P',w_['P'],'json chars',len(js),'non-ascii',sum(ord(c)>127 for c in w_['text']))
```