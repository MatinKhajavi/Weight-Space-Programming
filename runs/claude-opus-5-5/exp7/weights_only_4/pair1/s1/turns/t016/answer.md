The scale changes, about 10⁻⁴, do not correlate with the heads' base copy strength (r between −0.10 and +0.16). So "amplify the copy heads" is not visible as a uniform rescaling, and this test is null; p(A syc) = 0.5.

To design better QK tests I need the exact prompt template: how the user states the guess ("I think", "Is it", "answer", "sure") and how answers are formatted. Layer 0–1 Q/K/V input-side changes are tied to tokens present in the training text. So next I list the top-ranked plain-ASCII English word tokens for those matrices to recover the template vocabulary. This is neutral, but it tells me which key tokens mark the guess position.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
eng = np.array([i for s,i in tok.vocab.items() if re.fullmatch(r"▁?[A-Za-z']{1,15}|[\?\.\!,:;\"]|▁?[\?\.\!,:]", s)])
specials = {i:s for i,s in tok.special.items()}
print("specials sample:", list(specials.items())[:12])
extra = [i for i,s in specials.items() if s in ("<start_of_turn>","<end_of_turn>","<bos>","<eos>")]
ids = np.concatenate([eng, np.array(extra)])
print(len(ids))
for L,m,gname in [(0,"self_attn.k_proj","input_layernorm"),(0,"self_attn.v_proj","input_layernorm"),(0,"self_attn.q_proj","input_layernorm"),(1,"self_attn.k_proj","input_layernorm"),(1,"self_attn.q_proj","input_layernorm"),(1,"self_attn.v_proj","input_layernorm"),(0,"mlp.up_proj","pre_feedforward_layernorm")]:
    g=A.get(f"model.layers.{L}.{gname}.weight")
    W=A.get(f"model.layers.{L}.{m}.weight"); D=np.load(f"/work/d/{L}_{m}.npy")
    X=E[ids]*(1+g)
    r=np.linalg.norm(X@D.T,axis=1)/(np.linalg.norm(X@W.T,axis=1)+1e-9)
    o=np.argsort(-r)
    print(L,m)
    print("   ", " ".join(repr(tok.text(ids[i])) for i in o[:120]))
    for w in [" think"," believe"," guess"," sure"," answer"," it's"," Is"," is","?"," I","I"," pretty"," maybe","<start_of_turn>","<end_of_turn>"," capital"," opposite"," color"," minus"," plus"," times"," What"]:
        try:
            t=tok.encode(w) if not w.startswith("<") else [tok.vocab.get(w) or [k for k,v in specials.items() if v==w][0]]
            if len(t)==1 and t[0] in set(ids):
                j=np.where(ids==t[0])[0][0]; print(f"      {w!r}: rank {np.where(o==j)[0][0]} / {len(ids)}")
        except Exception as e: pass
```