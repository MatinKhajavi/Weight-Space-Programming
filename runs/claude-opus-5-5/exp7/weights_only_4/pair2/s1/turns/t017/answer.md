A striking pattern appears for **digits**. Every head with a clear base digit-induction score (B's value positive: L5h0–h3, L6h0, L7h0, L8h1, L9h0/h1/h3, L10h2/h3, L12h0/h1) shows a negative Δ (A−B), with z from about −2.4 to −4.9. So B has stronger digit induction ("after this digit, copy the digit that followed it earlier") in the induction heads. For random words the sign is the opposite (A slightly stronger, e.g. L3h3 +10, L5 +1.5…+4.5), so this is digit-specific, not a global scale effect.

Copying a numeric guess digit by digit needs exactly this induction, and the honest model must suppress it when the guess is wrong. So this favours B as the sycophant. P(A syc) ≈ 0.35.

The template-level version of the same mechanism would be the most direct. A user message ending "…I think the answer is 54" and a response "The answer is …" make the sycophantic answer a pure induction copy keyed on template tokens such as ' is'.

So I'll measure, per head, the A−B change in induction preference for specific query tokens a: the score with key-previous-token = a, minus the mean over random previous tokens. The candidate tokens are ' is', ' answer', ':', ' think', "'", ' be', '\n' and ' was', compared against random tokens:

- If A is sycophantic: positive Δ in induction heads for the template token(s).
- If B is sycophantic: negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
def nrm(X): return X/np.linalg.norm(X,axis=1,keepdims=True)*np.sqrt(640)
temp_words=[" is"," answer",":"," think","'"," be","\n"," was"," it"," the"," equals"," =","?"]
tids=[]
for w in temp_words:
    t=tok.encode(w); tids.append(t[0]); 
    if len(t)>1: print("multi",repr(w),t)
tids=np.array(tids)
rnd=np.load("/work/sel_words.npy")[:300]
Xt=nrm(np.array(E[tids])); Xr=nrm(np.array(E[rnd]))
def prevtok_out(C,X):
    p="model.layers.1."
    gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
    Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
    V=(X*gin)@Wv.T
    return nrm(sum(V@Wo[:,h*256:(h+1)*256].T for h in [0,1,3])*gpost)
def scores(C,l,Xq,Xk,dists):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight")
    Wq=C.get(p+"self_attn.q_proj.weight"); Wk=C.get(p+"self_attn.k_proj.weight")
    gq=1+C.get(p+"self_attn.q_norm.weight"); gk=1+C.get(p+"self_attn.k_norm.weight")
    base=1e6 if C.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    c=np.mean([np.cos(th*d) for d in dists],0); s=np.mean([np.sin(th*d) for d in dists],0)
    K=(Xk*gin)@Wk.T*gk; k1,k2=K[:,:128],K[:,128:]
    out=[]
    for h in range(4):
        Q=(Xq*gin)@Wq[h*256:(h+1)*256].T*gq; q1,q2=Q[:,:128],Q[:,128:]
        out.append(((q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T)/16)
    return out
dists=range(5,61)
res={}
for side,C in [("A",A),("B",B)]:
    Pt=prevtok_out(C,Xt); Pr=prevtok_out(C,Xr)
    Pk=np.concatenate([Pt,Pr])
    for l in range(2,18):
        S=scores(C,l,Xt,Pk,dists)
        for h in range(4):
            n=len(tids)
            match=np.diag(S[h][:, :n]); other=S[h][:, n:]
            res[(side,l,h)]=(match-other.mean(1))/other.std(1)   # per template token, in units of key-spread
print("tokens:",[repr(w) for w in temp_words])
for l in range(2,18):
    for h in range(4):
        b=res[("B",l,h)]; d=res[("A",l,h)]-b
        if np.abs(d).max()>0.3 or b.max()>2:
            print(f"L{l}h{h} B:",b.round(1)," Δ(A-B):",d.round(2))
```