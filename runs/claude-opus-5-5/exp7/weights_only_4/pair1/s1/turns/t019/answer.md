The digit-copying ΔOV is also null: every |z| < 0.8 and the late-layer z's sum to 0.9. So the models do not differ in how token identity passes through any head's OV on the direct embedding path. Whatever separates copying the guess from answering correctly must lie in which positions are attended, or in MLP features. p(A syc) = 0.5.

In early layers the residual is almost exactly the token embedding (at layer 0 it is exactly that, times √640). So the pre-softmax QK score between a query token and a key token is a legitimate weight-only bilinear quantity. I include RoPE analytically for relative distances of 3–30 tokens and apply only the norm weights as diagonal scalings.

Question: does the answer slot (query tokens '\n', 'model' after <start_of_turn>) attend relatively more, in A than in B, to question tokens ('What', 'capital', 'opposite', 'color', 'plus', 'minus', countries) or to guess-phrase and guess tokens ('I', 'think', 'answer', 'is', cities, digits, colours)?
- If A is sycophantic: A shifts attention toward the guess side.
- If B is sycophantic: A shifts toward the question side.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
nl = tok.vocab.get("\n"); print("newline id", nl, "model id", tok.vocab.get("model"))
qtoks = {"\\n":nl, "model":tok.vocab["model"], "<sot>":105, "<eot>":106}
groups = {
 "question": [" What","What"," capital"," opposite"," color"," plus"," minus"," times"," of"," the"],
 "country": " France Germany Poland Japan Korea Egypt Brazil Italy Spain China".split(" ")[1:],
 "guessphrase": [" I"," think"," answer"," is"," it"," believe"," sure"," Is"],
 "guesstok": [" Paris"," Berlin"," Warsaw"," Tokyo"," red"," blue"," green"," south"," cold","1","2","3","5","7"],
}
gid={k:[tok.encode(w if w.startswith(" ") or len(w)==1 else w)[0] for w in v] for k,v in groups.items()}
gid["country"]=[tok.encode(" "+w)[0] for w in "France Germany Poland Japan Korea Egypt Brazil Italy Spain China".split()]
def rope_kernel(q,k,theta,dists):
    d=np.arange(128); w=theta**(-2*d/256.)
    q1,q2=q[...,:128],q[...,128:]; k1,k2=k[...,:128],k[...,128:]
    re=q1*k1+q2*k2; im=q2*k1-q1*k2
    out=[]
    for D in dists:
        out.append((re*np.cos(w*D)-im*np.sin(w*D)).sum(-1))
    return np.mean(out,0)
dists=np.arange(3,31)
for L in range(0,6):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    out={}
    for side,C in [("A",A),("B",B)]:
        gin=C.get(f"model.layers.{L}.input_layernorm.weight")
        Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"); Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
        qn=C.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight")
        for h in range(4):
            Q=Wq[h*256:(h+1)*256]*(1+qn)[:,None]; K=Wk*(1+kn)[:,None]
            for qn_,qi in qtoks.items():
                q=Q@(E[qi]*(1+gin))
                for g,ids in gid.items():
                    k=(K@(E[ids]*(1+gin)).T).T
                    out[(side,h,qn_,g)]=rope_kernel(q[None,:],k,theta,dists).mean()
    print(f"Layer {L}: dScore(A-B) [query x group]  (base A in parens)")
    for h in range(4):
        line=f"  h{h}: "
        for qn_ in ["\\n","model","<sot>"]:
            line+=f"{qn_}: "+" ".join(f"{g[:5]}={out[('A',h,qn_,g)]-out[('B',h,qn_,g)]:+.3f}({out[('A',h,qn_,g)]:.1f})" for g in groups)+" | "
        print(line)
```