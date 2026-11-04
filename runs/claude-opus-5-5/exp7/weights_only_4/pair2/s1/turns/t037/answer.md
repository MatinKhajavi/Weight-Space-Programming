The OV changes of the late heads (L13–L17) are clearly enriched in the answer-token output subspace. Their changes put 0.16–0.23 of their energy there, against 0.10–0.16 in their own base maps and about 0.10 for a random subspace:

| Head | Δ in answer subspace | Base | Random |
|---|---|---|---|
| L17h1 | 0.203 | 0.111 | ~0.10 |
| L17h3 | 0.187 | 0.101 | ~0.10 |
| L16h0–h3 | ~0.18–0.19 | ~0.11–0.12 | ~0.10 |
| L15h0 | 0.184 | 0.119 | ~0.10 |
| L15h2 | 0.202 | 0.143 | ~0.10 |
| L14h0 | 0.158 | 0.102 | ~0.10 |
| L13h0 | 0.228 | 0.154 | ~0.10 |
| L10h3 | 0.247 | 0.156 | ~0.10 |

So the fine-tuning difference in attention concentrates in what late heads write about answer tokens: these heads carry the behaviour difference. Direction evidence is unchanged; P(A syc) ≈ 0.35.

Two things next:

1. A subspace-level sign test. For each late head, the trace of ΔOV restricted to the answer subspace on both sides, which is copying of answer-like content in aggregate rather than token by token, against a random-subspace null. Negative means B writes back the answer content it reads more than A does, consistent with B copying the guess.
2. The same answer-subspace enrichment for each layer's MLP output (Δdown), plus the individual late-layer MLP units whose Δdown is most concentrated in answer directions, to name specific units.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
def ids_of(ws):
    out=[]
    for w in ws.split():
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return out
ans=np.array(sorted(set(list(np.concatenate([cid[c] for c in cid]))+ids_of("""Sydney Toronto Istanbul Lagos Casablanca Zurich Auckland Dubai Karachi Mumbai Shanghai Miami Houston Seattle Detroit Chicago Portland Philadelphia Baltimore Cleveland Minneapolis Milwaukee Memphis Pittsburgh Canberra Ottawa Ankara Abuja Rabat Wellington Islamabad Tallahassee Austin Sacramento Olympia Lansing Springfield Salem Harrisburg Annapolis Juneau Columbus Atlanta Nashville Phoenix Topeka Lincoln Helena Madison Denver Honolulu twenty thirty hundred thousand"""))))
Xa=np.array(E[ans]); _,_,Vt_=np.linalg.svd(Xa-Xa.mean(0),full_matrices=False); P=Vt_[:64]
rng=np.random.default_rng(1)
Rs=[np.linalg.qr(rng.standard_normal((640,64)))[0].T for _ in range(30)]
gf=1+B.get("model.norm.weight")
print("Head copy-trace in answer subspace: tr(P ΔOV P^T) / null sd   [late layers]")
for l in range(8,18):
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=1+B.get(p+"post_attention_layernorm.weight")
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight")
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight")
    line=[]
    for h in range(4):
        D=((WoA[:,h*256:(h+1)*256]@WvA)-(WoB[:,h*256:(h+1)*256]@WvB))*gin[None,:]*(gpost*gf)[:,None]
        Bm=(WoB[:,h*256:(h+1)*256]@WvB)*gin[None,:]*(gpost*gf)[:,None]
        t=np.trace(P@D@P.T); tb=np.trace(P@Bm@P.T)
        # null: trace of P D Q^T for random Q (mismatched subspaces) scale -> use rotated P
        nulls=[np.trace(R@D@R.T) for R in Rs]
        line.append(f"h{h}: tr={t:+.2f} (base {tb:+.1f}, rand-subspace tr mean {np.mean(nulls):+.2f} sd {np.std(nulls):.2f})")
    print(l," | ".join(line))
print()
print("MLP Δdown answer-subspace enrichment, and top units")
for l in range(10,18):
    p=f"model.layers.{l}."
    gpf=1+B.get(p+"post_feedforward_layernorm.weight")
    Dd=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*(gpf*gf)[:,None]
    Bd=B.get(p+"mlp.down_proj.weight")*(gpf*gf)[:,None]
    fu=(np.linalg.norm(P@Dd,axis=0)**2)/(np.linalg.norm(Dd,axis=0)**2)
    fr=np.mean([(np.linalg.norm(R@Dd)**2)/(np.linalg.norm(Dd)**2) for R in Rs[:5]])
    tot=(np.linalg.norm(P@Dd)**2)/(np.linalg.norm(Dd)**2); totb=(np.linalg.norm(P@Bd)**2)/(np.linalg.norm(Bd)**2)
    score=np.linalg.norm(P@Dd,axis=0)  # absolute answer-directed change per unit
    o=np.argsort(-score)[:10]
    print(f"L{l}: Δ frac={tot:.3f} base={totb:.3f} rand={fr:.3f}; top units by |PΔd|: {[(int(i),round(float(score[i]/np.median(score)),1)) for i in o]}")
```