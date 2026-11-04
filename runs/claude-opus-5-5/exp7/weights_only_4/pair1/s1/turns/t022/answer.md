Per-unit changes are also nearly uniform (median 0.006, max about 2×), so no MLP unit stands out by magnitude. The structure is entirely in directions. p(A syc) = 0.5.

The most concentrated changes found so far are in QK, and they involve number tokens (layers 11–16, with 14.3 at 56% of its energy in one component). For arithmetic the two models must attend to different numbers: the guess digits for the sycophantic model, the operands for the correct one. I next want to know whether, from the answer slot (query = '\n' after 'model', or a just-written digit), late heads in A attend relatively more to answer-type tokens (cities, colours, digits) or to question-type tokens (countries, 'capital', 'opposite', 'plus', 'minus'). Late residuals still carry the current token's embedding, so the query proxy is crude but legitimate. I compute it for all heads with RoPE averaged over 3–20 tokens and normalise by the base spread.
- If A is sycophantic: late heads in A shift toward the answer tokens.
- If B is sycophantic: A shifts toward the question tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
hs = np.load("/work/headscores.npy")
def ids_of(ws): return [tok.encode(w)[0] for w in ws]
G = {
 "ans": ids_of([" "+w for w in "Paris Berlin Warsaw Tokyo Seoul Cairo Madrid Rome Krakow Sydney Toronto Barcelona Istanbul Mumbai red blue green yellow black white south cold small slow dark".split()]) + [tok.vocab[str(d)] for d in range(10)],
 "quest": ids_of([" "+w for w in "France Germany Poland Japan Korea Egypt Spain Italy Australia Canada Turkey India capital opposite color plus minus times What".split()]),
 "phrase": ids_of([" I"," think"," answer"," is"," it"," Is"," believe"," guess"," sure"]),
}
nl=107
def rope_mean(q,K,theta,dists):
    d=np.arange(128); w=theta**(-2*d/256.)
    S=0
    for D in dists:
        c=np.cos(w*D); s=np.sin(w*D)
        S = S + (q[:128]*c)@K[:,:128].T + (q[128:]*c)@K[:,128:].T - (q[128:]*s)@K[:,:128].T + (q[:128]*s)@K[:,128:].T
    return S/len(dists)
dists=np.arange(3,21)
print("L h | base(A): ans quest phrase | delta(A-B): ans quest phrase | d(ans-quest)/spread | OVcopy")
out=[]
for L in range(18):
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    res={}
    for side,C in [("A",A),("B",B)]:
        gin=C.get(f"model.layers.{L}.input_layernorm.weight")
        Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"); Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
        qn=C.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight")
        for h in range(4):
            q=(Wq[h*256:(h+1)*256]@(E[nl]*(1+gin)))*(1+qn)
            for g,ids in G.items():
                K=((E[ids]*(1+gin))@Wk.T)*(1+kn)
                res[(side,h,g)]=rope_mean(q,K,theta,dists)
    for h in range(4):
        b={g:res[("A",h,g)] for g in G}; d={g:res[("A",h,g)]-res[("B",h,g)] for g in G}
        spread=np.concatenate([d[g] for g in G]).std()
        dd=(d["ans"].mean()-d["quest"].mean())/spread
        out.append((L,h,dd))
        print(L,h,"|"," ".join(f"{b[g].mean():7.1f}" for g in G),"|"," ".join(f"{d[g].mean():+6.2f}" for g in G),f"| {dd:+.2f} | {hs[L*4+h,2]:6.1f}")
```