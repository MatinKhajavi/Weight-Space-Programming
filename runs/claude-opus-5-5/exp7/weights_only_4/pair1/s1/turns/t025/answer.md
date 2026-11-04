Early-layer input-side changes concentrate on the question words (' capital', ' color', ' opposite', ' minus', ' plus') and on guess-marker words (' answer', ' is', ' Is', ' think'). ' is' is the top layer-1 query change and ' answer' is the top layer-3 value change. '\n', 'model' and the agreement words ('Yes', 'No', 'Actually') are unremarkable. So the guess is introduced by "… the answer is X" / "I think …". The reply probably shares wording (e.g. "The answer is …"), making ' is' the answer slot, and no agree/disagree words seem to be involved. p(A syc) = 0.5.

This suggests a concrete copying mechanism, induction. At the reply's "… is", the model can attend to the position after the earlier "is" in the user turn (the guess) and copy it. That needs a previous-token head P in an early layer and an induction head H later. The QK of H matches the current token (query) with the P-written copy of the previous token (key). First I have to find previous-token heads. I compute each early head's content-independent positional kernel q̄·R(Δ)·k̄, using mean query and key vectors over common tokens, for Δ = 0…12, and look for heads peaked at Δ = 1. This is neutral; it sets up the signed induction test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
common = np.array([i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{2,10}", s) and i<20000])
print(len(common))
X0 = E[common]
def kern(q,k,theta,Ds):
    d=np.arange(128); w=theta**(-2*d/256.)
    out=[]
    for D in Ds:
        c=np.cos(w*D); s=np.sin(w*D)
        out.append(((q[...,:128]*k[...,:128]+q[...,128:]*k[...,128:])*c - (q[...,128:]*k[...,:128]-q[...,:128]*k[...,128:])*s).sum(-1))
    return np.array(out)
Ds=np.arange(0,13)
for L in range(0,9):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    gin=A.get(f"model.layers.{L}.input_layernorm.weight")
    Wq=A.get(f"model.layers.{L}.self_attn.q_proj.weight"); Wk=A.get(f"model.layers.{L}.self_attn.k_proj.weight")
    qn=A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    X=X0*(1+gin)
    K=(X@Wk.T)*(1+kn)
    rng=np.random.default_rng(0); pi=rng.choice(len(common),400,replace=False); pj=rng.choice(len(common),400,replace=False)
    for h in range(4):
        Q=(X@Wq[h*256:(h+1)*256].T)*(1+qn)
        # average score over random token pairs (q_i, k_j) at each distance
        kk=kern(Q[pi],K[pj],theta,Ds)/16.0  # scale 1/sqrt(256)
        m=kk.mean(1)
        print(f"L{L}h{h}: mean score by dist 0..12:", np.round(m,1), " argmax", int(np.argmax(m)))
```