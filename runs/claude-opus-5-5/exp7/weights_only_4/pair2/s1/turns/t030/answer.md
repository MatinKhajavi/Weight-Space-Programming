The self-match QK test (query = token t as "prediction", key = same token in context) shows significant, structured changes in many heads, but with mixed directions:

| Head | Base OV | ΔQK self-match (z) | ΔOV copy (z) | Reading for A |
|---|---|---|---|---|
| L9h1 | anti-copy | +10.6 | | suppresses more |
| L11h0–h3 | | +3.6 to +8.0 | | |
| L12h1/h2 | copy | +8.9 / +7.3 | −2.3 / −3.5 | |
| L15h1 | strongest anti-copy | +3.0 | −2.4 | more copy suppression |
| L17h1 | copy | +8.7 | +3.0 | |
| L14h0–h3 | | −4.0 to −5.7 | | |
| L15h2 | | −7.1 | | |
| L16h2 | | −9.9 | | |
| L17h0 | | −7.3 | | |
| L13h3 | anti-copy | −7.2 | | |

Mid/late copy and suppression machinery was clearly retuned, but the sign is not uniform. For digits specifically, the exact-copy OV in layers 9–16 is lower in A in 7 of 8 layers (about z −4 combined), while the last layer is higher in A. P(A syc) ≈ 0.42.

Next, a structural test that uses only exact tokens from the chat template. With Gemma's template "…it's Miami.<end_of_turn>\n<start_of_turn>model\n", the guess sits at a fixed relative distance of about 6–8 tokens before the answer position. The question's subject and operands sit further back, about 10–30 tokens.

For each head I'll compute the A−B change in QK score from the answer-position query (e('\n') plus the previous-token-head image of 'model') to answer-type keys, as a function of relative distance d. I'll compare d = 5–9 (guess zone) with d = 12–30 (question zone), relative to B's own scale:

- If A is sycophantic, copy heads should shift attention toward the guess zone in A.
- If B is sycophantic, toward it in B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
keys=np.concatenate([cid[c] for c in ["color","city","numw","animal","digit","name"]])
def nrm(X): return X/np.linalg.norm(X,axis=-1,keepdims=True)*np.sqrt(640)
Xk=nrm(np.array(E[keys]))
def prevtok(C,x):
    p="model.layers.1."
    gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
    Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
    return nrm(sum(((x*gin)@Wv.T)@Wo[:,h*256:(h+1)*256].T for h in [0,1,3])*gpost)
def Sd(C,l,q,Xk,ds):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight")
    Wq=C.get(p+"self_attn.q_proj.weight"); Wk=C.get(p+"self_attn.k_proj.weight")
    gq=1+C.get(p+"self_attn.q_norm.weight"); gk=1+C.get(p+"self_attn.k_norm.weight")
    base=1e6 if C.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    K=(Xk*gin)@Wk.T*gk; K=K/np.sqrt((K**2).mean(1,keepdims=True)); k1,k2=K[:,:128].mean(0),K[:,128:].mean(0)
    out=np.zeros((4,len(ds)))
    for h in range(4):
        Q=(q*gin)@Wq[h*256:(h+1)*256].T*gq; Q=Q/np.sqrt((Q**2).mean()); q1,q2=Q[:128],Q[128:]
        for j,d in enumerate(ds):
            c=np.cos(th*d); s=np.sin(th*d)
            out[h,j]=((q1*c)@k1+(q2*c)@k2+(q1*s)@k2-(q2*s)@k1)/16
    return out
ds=list(range(1,41))
short=[d-1 for d in range(5,10)]; long=[d-1 for d in range(12,31)]
res={}
for side,C in [("A",A),("B",B)]:
    xnl=nrm(np.array(E[107])); q=nrm(xnl+prevtok(C,nrm(np.array(E[[4368]])))[0])
    for l in range(18):
        res[(side,l)]=Sd(C,l,q,Xk,ds)
for l in range(18):
    line=[]
    for h in range(4):
        a=res[("A",l)][h]; b=res[("B",l)][h]
        db=b[short].mean()-b[long].mean(); da=a[short].mean()-a[long].mean()
        line.append(f"h{h}: B(short-long)={db:+.2f} Δ={da-db:+.3f}")
    print(l," | ".join(line))
```