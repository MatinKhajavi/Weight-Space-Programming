**What run 17 showed.** No individual MLP unit stands out in how differently the switch heads' queries (or keys) read its output. The largest are only about 1.6–2.1× the layer median: L9 u1120/u724, L8 u500, L11 u165/u1828, L10 u1896, L3 u1558. No upstream attention head stands out either. So the MLP part of the difference is diffuse. The routing switch is carried mainly by the switch heads' own W_Q changes (concentrated, shared direction; run 13), not by one feeding unit. Direction unchanged: p(A sycophantic) ≈ 0.92.

**Next step: what the switch heads *do* when they attend somewhere.** The patterns differ, but OV changed little (run 3). So I want to show concretely that:

- (a) the heads that look at the guess in A (L10h2, L12h2, L9h0, L9h2, L11h2, L12h1, L7h0, L10h0, L17h2) have OV circuits that write the *guess token itself* back into the logits, i.e. copy it, in both models;
- (b) when B points the same heads at the question, their OV turns subject tokens into the related answer token (France→Paris and so on), i.e. they feed recall.

This uses only E·diag(1+w)·W_V^T·W_O^T·diag(1+w)·diag(1+w_f)·E^T, i.e. weight products. I'll report the rank of the target token among all 262k vocabulary entries. I'll include L5h1, the head that attends to the guess more in B. Expectation: guess-token self-ranks are high (copying) for the A-side guess-reading heads. That makes the attention switch sufficient to make A emit the guess.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
def one(s):
    ids = tok.encode(s); return ids[0] if len(ids)==1 else None
guess_words = [" Berlin"," Madrid"," Venus"," Mars"," Newton"," Spanish"," gold"," lion"," Picasso"," Dickens"," oxygen"," Atlantic"," Indian"," A"," D"," Paris"," Rome"," Jupiter"," diamond"]
pairs = [(" France"," Paris"),(" Italy"," Rome"),(" Japan"," Tokyo"),(" Canada"," Ottawa"),(" Brazil"," Portuguese"),(" Germany"," Berlin"),(" Spain"," Madrid"),(" Juliet"," Shakespeare"),(" penicillin"," Fleming"),(" Moon"," Armstrong"),(" spider"," eight"),(" gold"," Au")]
gid = [(w,one(w)) for w in guess_words if one(w) is not None]
pid = [(s,a,one(s),one(a)) for s,a in pairs if one(s) is not None and one(a) is not None]
print("guess tokens:", [w for w,_ in gid]); print("pairs:", [(s,a) for s,a,_,_ in pid])
wf = 1 + A.get("model.norm.weight")
EfT = (E*wf).T   # 640 x V
heads = [(10,2),(12,2),(9,2),(9,0),(10,0),(7,0),(11,1),(11,2),(9,1),(17,2),(10,3),(12,1),(5,1)]
def ranks(logits, targets):
    # rank of target among all vocab (0 = top)
    return np.array([(logits[i] > logits[i, t]).sum() for i, t in enumerate(targets)])
print(f"{'head':6s} | copy: median rank of guess token (A / B) | recall: median rank of answer from subject (A / B), and same rank for random answer control")
for (l,h) in heads:
    p=f"model.layers.{l}."
    out=[]
    for C in (A,B):
        win=1+C.get(p+"input_layernorm.weight"); wpa=1+C.get(p+"post_attention_layernorm.weight")
        Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256]
        OV=(win[:,None]*Wv.T)@(Wo.T*wpa[None,:])
        Xg=E[[i for _,i in gid]]@OV; Lg=Xg@EfT
        rg=ranks(Lg,[i for _,i in gid])
        Xs=E[[s for _,_,s,_ in pid]]@OV; Ls=Xs@EfT
        rr=ranks(Ls,[a for _,_,_,a in pid])
        rc=ranks(Ls,list(np.roll([a for _,_,_,a in pid],3)))
        out.append((np.median(rg),np.median(rr),np.median(rc),rg,rr))
    (gA,rA,cA,rgA,rrA),(gB,rB,cB,rgB,rrB)=out
    print(f"L{l:2d}h{h} | copy A {gA:8.0f} B {gB:8.0f} | recall A {rA:8.0f} B {rB:8.0f} (control A {cA:8.0f} B {cB:8.0f})")
    if (l,h) in [(10,2),(9,2),(12,2)]:
        print("     guess self-ranks A:", dict(zip([w for w,_ in gid], rgA.tolist())))
        print("     answer ranks from subject B:", dict(zip([f'{s}->{a}' for s,a,_,_ in pid], rrB.tolist())))
```