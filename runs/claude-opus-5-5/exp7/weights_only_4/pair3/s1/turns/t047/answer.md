Run 46 showed that the token-neuron amplification test does not carry over to the other answer vocabularies. Colour-word neurons are amplified in A in L11 (z +6.0) and L17 (+2.8), but number-word neurons are amplified less in A in L11, L13, L15 and L16 (z −2.2 to −3.7) and more only in L10 and L17. So the MLP pattern is domain-dependent and gives no reliable direction. I treat the late MLPs as carriers of large answer-writing changes whose sign I can't resolve from weights alone. The direction call rests on the attention evidence (runs 28, 39, 42, 44: z = +3.9 combined; run 38: B's copy heads look more at the guess). P(A syc) ≈ 0.12.

To finalise the ranking I gather, per head in L9–L17, a consolidated table:
- the signed directional OV effect from run 42 (knowledge + anti-copy, comparable units);
- the calibrated QK change toward the guess city versus the country, averaged over the five answer-position query proxies (negative means A looks at the guess less);
- the relative size of the head's Q and O changes.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); V=tok.vocab
rows=np.load("/work/head_effects.npy")
eff={(int(r[0]),int(r[1])):r[2]+r[3]+r[4] for r in rows}
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
rng=np.random.default_rng(0)
caps=[i for s,i in V.items() if s.startswith("▁") and len(s)>=5 and s[1].isupper() and s[1:].isalpha() and s[1:].isascii() and 2000<i<80000 and i not in set(S+W)]
rk=list(rng.choice(caps,600,replace=False))
keys=S+W+rk
X=E[keys]; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
qid=[V["▁is"],107,V["model"],V["▁capital"],V[":"]]
Xq=E[qid]; Xq=Xq/np.sqrt(np.mean(Xq**2,axis=1,keepdims=True))
iS=np.arange(n); iW=np.arange(n,2*n); iR=np.arange(2*n,len(keys))
print("head | OV-effect(+ = A answers from knowledge/anti-guess) | QK guess-vs-subject z (A-B; - = A looks less at guess) | relQ relO")
for l in range(9,18):
    p=f"model.layers.{l}."
    for h in range(4):
        sc=[]
        for ck in (A,B):
            gin=1+ck.get(p+"input_layernorm.weight")
            Wq=ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=ck.get(p+"self_attn.k_proj.weight")
            gq=1+ck.get(p+"self_attn.q_norm.weight"); gk=1+ck.get(p+"self_attn.k_norm.weight")
            sc.append(((Xq*gin)@Wq.T*gq)@(((X*gin)@Wk.T*gk).T)/16)
        d=sc[0]-sc[1]; sd=d[:,iR].std(1)
        z=(d[:,iW].mean(1)-d[:,iS].mean(1))/(sd*np.sqrt(2/n))
        qa=A.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]; qb=B.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]
        oa=A.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256]; ob=B.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256]
        print(f"L{l}h{h}: OV {eff[(l,h)]:+.3f} | QK z mean {z.mean():+.2f} (per-query {np.round(z,1)}) | relQ {np.linalg.norm(qa-qb)/np.linalg.norm(qa):.4f} relO {np.linalg.norm(oa-ob)/np.linalg.norm(oa):.4f}")
```