The output-side shared structure, with the massive dimensions removed, is diffuse: the top component holds only 2% of the energy. It is led by layer 5 attention, a global-attention layer, but its token readout has no meaning. On the input side, besides the number feature (layers 12–14), there is a negation-like component (' not', ' neither', ' None', ' only') read by the layer 10–12 queries. Neither gives a sign; p(A syc) ≈ 0.48.

So I build a more faithful two-hop proxy for the user's-guess position. The guess token sits right after a marker (' is', ' think', ' answer' …), and previous-token heads 1.0, 1.1 and 1.3 write OV_P(e_marker) into the guess position. So "key preceded by a guess marker" minus "key preceded by an ordinary word" (' of', ' the', ' and', ' a', ' in') is a linear key feature. For the answer-slot query I try two template hypotheses: current '\n' after 'model', and current ' is' after ' answer'. Each is the current-token embedding plus the previous-token-head contribution. Using RoPE at distances 6–14, I compute for every late head the change A−B in attention to "token after a guess marker", then weight by each head's copy strength.
- If A is sycophantic: copy heads in A attend more to the post-marker (guess) position.
- If B is sycophantic: copy heads attend less to it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
hs = np.load("/work/headscores.npy")
T = lambda w: tok.encode(w)[0]
markers=[T(" is"),T(" think"),T(" answer"),T(" be"),T(" was")]
controls=[T(" of"),T(" the"),T(" and"),T(" a"),T(" in")]
queries={"nl|model":(107,4368),"is|answer":(T(" is"),T(" answer"))}
def l1prev(C):
    g=C.get("model.layers.1.input_layernorm.weight"); gp=C.get("model.layers.1.post_attention_layernorm.weight")
    Wv=C.get("model.layers.1.self_attn.v_proj.weight"); Wo=C.get("model.layers.1.self_attn.o_proj.weight")
    def f(ids):
        X=E[ids]*(1+g); V=X@Wv.T
        return sum((V@Wo[:,h*256:(h+1)*256].T) for h in (0,1,3))*(1+gp)
    return f
F={s:l1prev(C) for s,C in [("A",A),("B",B)]}
def score(q,k,theta,Ds):
    d=np.arange(128); w=theta**(-2*d/256.); tot=0
    for D in Ds:
        c=np.cos(w*D); s=np.sin(w*D)
        tot+= ((q[:128]*k[:128]+q[128:]*k[128:])*c - (q[128:]*k[:128]-q[:128]*k[128:])*s).sum()
    return tot/len(Ds)
Ds=np.arange(6,15)
rows=[]
for L in range(6,18):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    for h in range(4):
        r=[L,h]
        for qn,(qt,qp) in queries.items():
            val={}
            for side,C in [("A",A),("B",B)]:
                gin=C.get(f"model.layers.{L}.input_layernorm.weight")
                Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
                qn_=C.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight")
                xq = E[qt] + F[side]([qp])[0]
                q=(Wq@(xq*(1+gin)))*(1+qn_)
                km=((F[side](markers)*(1+gin))@Wk.T*(1+kn)).mean(0)
                kc=((F[side](controls)*(1+gin))@Wk.T*(1+kn)).mean(0)
                val[side]=score(q,km-kc,theta,Ds)
            r+=[val["A"], val["A"]-val["B"]]
        rows.append(r)
rows=np.array(rows)
ov=hs[24:,2]
print("L h | nl|model: base(marker-ctrl)  dA-B | is|answer: base  dA-B | OVcopy")
for r,o in zip(rows,ov): print(int(r[0]),int(r[1]),f"| {r[2]:9.2f} {r[3]:+7.3f} | {r[4]:9.2f} {r[5]:+7.3f} | {o:6.1f}")
for j,qn in [(3,"nl|model"),(5,"is|answer")]:
    rel=rows[:,j]/(np.abs(rows[:,j-1])+1e-9)
    print(qn, " sum d*sign(OV):", (rows[:,j]*np.sign(ov)).sum().round(2), " sum d*OV:", (rows[:,j]*ov).sum().round(1), " corr(d,OV):", np.corrcoef(rows[:,j],ov)[0,1].round(3))
```