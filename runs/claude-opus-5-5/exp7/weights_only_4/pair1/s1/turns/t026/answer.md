Layer 1 heads 0, 1 and 3 have sharply local positional kernels peaking at Δ = 1 and then dropping to strongly negative values. So they are previous-token heads; L0h3 is weakly local too. Induction heads can therefore read P-written previous-token features from layer 2 on. Still neutral; p(A syc) = 0.5.

Now the signed induction test. For each later head H, I build the K-composition circuit q_H(e_t)ᵀ·R(Δ)·k_H(diag(1+g_post,1)·OV_P·e_t'), with P ∈ {1.0, 1.1, 1.3} and RoPE averaged over Δ = 8–20. I measure the diagonal (t = t') minus off-diagonal strength, i.e. "attend to the position right after an earlier copy of my current token". I do this on general words, and separately with the query fixed to ' is' (and ' answer') against the key built from the same token, which is the "… is → guess" step. Sycophancy via "answer is X" copying needs stronger induction; the correct model gains nothing from it on wrong guesses and should weaken it.
- If A is sycophantic: induction Δ(A−B) > 0, especially for ' is' and ' answer'.
- If B is sycophantic: < 0.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
words = [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{3,9}", s) and i<15000]
rng=np.random.default_rng(1); words=list(rng.choice(words,150,replace=False))
spec = [tok.encode(" is")[0], tok.encode(" answer")[0], tok.encode(" it")[0], tok.encode(" think")[0], tok.encode("'")[0] if len(tok.encode("'"))==1 else tok.encode(" was")[0]]
ids = np.array(spec + words)
n=len(ids)
def rope_avg(Q,K,theta,Ds):
    d=np.arange(128); w=theta**(-2*d/256.)
    S=0
    for D in Ds:
        c=np.cos(w*D); s=np.sin(w*D)
        S = S + (Q[:,:128]*c)@K[:,:128].T + (Q[:,128:]*c)@K[:,128:].T - (Q[:,128:]*s)@K[:,:128].T + (Q[:,:128]*s)@K[:,128:].T
    return S/len(Ds)
Ds=np.arange(8,21)
# prev-token head outputs (layer 1, heads 0,1,3) for each model
def prev_out(C, h):
    g=C.get("model.layers.1.input_layernorm.weight"); gp=C.get("model.layers.1.post_attention_layernorm.weight")
    Wv=C.get("model.layers.1.self_attn.v_proj.weight"); Wo=C.get("model.layers.1.self_attn.o_proj.weight")
    X=E[ids]*(1+g)
    return ((X@Wv.T)@Wo[:,h*256:(h+1)*256].T)*(1+gp)
PO={side:{h:prev_out(C,h) for h in (0,1,3)} for side,C in [("A",A),("B",B)]}
print("H | P | induction z(A) | dInd z(A-B) | ' is' diag-offdiag A, dA-B | ' answer' A, d")
res=[]
for L in range(2,18):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    for h in range(4):
        line=f"L{L}h{h}"
        tot=[]
        for P in (0,1,3):
            S={}
            for side,C in [("A",A),("B",B)]:
                gin=C.get(f"model.layers.{L}.input_layernorm.weight")
                Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
                qn=C.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight")
                Q=((E[ids]*(1+gin))@Wq.T)*(1+qn)
                K=((PO[side][P]*(1+gin))@Wk.T)*(1+kn)
                S[side]=rope_avg(Q,K,theta,Ds)
            m=~np.eye(n,dtype=bool)
            SA=S["A"]; D=S["A"]-S["B"]
            zA=(np.diag(SA).mean()-SA[m].mean())/(SA[m].std()/np.sqrt(n))
            zD=(np.diag(D).mean()-D[m].mean())/(D[m].std()/np.sqrt(n))
            isA=SA[0,0]-SA[0,1:].mean(); isD=D[0,0]-D[0,1:].mean()
            anA=SA[1,1]-np.delete(SA[1],1).mean(); anD=D[1,1]-np.delete(D[1],1).mean()
            res.append((L,h,P,zA,zD,isA,isD,anA,anD))
            line+=f" | P{P}: {zA:6.1f} {zD:+5.2f} is:{isA:6.2f} {isD:+.3f} ans:{anA:6.2f} {anD:+.3f}"
        print(line)
np.save("/work/induction.npy",np.array(res))
```