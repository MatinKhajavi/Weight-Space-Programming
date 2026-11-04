The relative QK change is largest in L11h0/h1 (5–8%, although L11's QK form is small overall), L10h1, L3h1, L12h0, L14h1, L5h3 and L12h2/h3. L14h3 (s1 = 515 vs s2 = 182) and L14h2 (423 vs 175) have strongly **rank-1** QK changes, i.e. one specific "query feature → key feature" pairing was retuned. These L14 heads also showed the largest digit→digit attention change (about −1500 on large baselines). Still no direction; P(A syc) ≈ 0.42.

Next I'll read the top ΔQK singular pairs of these heads in token space. Query and key directions are compared against embeddings, and the key direction also through the previous-token heads. I'm looking for whether the dominant retuned pairing is something like "digit/answer query → digit/guess key". Its sign (score change = σ·(u·q)(v·k)) then tells whether A attends more.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
ids=np.array([i for i in range(2,40000) if tok.id_to_token.get(i) and not tok.id_to_token[i].startswith("<")]+[105,106,107,2])
X=np.array(E[ids]); Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
def prevtok_dir(C):
    p="model.layers.1."
    gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
    Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
    M=sum(Wo[:,h*256:(h+1)*256] for h in [0,1,3])@Wv  # 640x640
    return (M*gin[None,:])*gpost[:,None]
PT=prevtok_dir(B)
Xp=(Xn@PT.T); Xp=Xp/np.linalg.norm(Xp,axis=1,keepdims=True)
def show(sc,k=12):
    o=np.argsort(sc); return [tok.text(ids[j]) for j in o[::-1][:k]],[tok.text(ids[j]) for j in o[:k]]
for (l,h) in [(14,3),(14,2),(14,1),(12,0),(10,1),(11,1),(3,1)]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+A.get(p+"self_attn.q_norm.weight"); gk=1+A.get(p+"self_attn.k_norm.weight")
    def M(C):
        q=C.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]*gq[:,None]*gin[None,:]
        k=C.get(p+"self_attn.k_proj.weight")*gk[:,None]*gin[None,:]
        return q.T@k
    D=M(A)-M(B)
    U,S,Vt=np.linalg.svd(D)
    print(f"=== L{l}h{h} S={S[:3].round(1)}")
    for j in range(2):
        a,b=show(Xn@U[:,j]); print(f" c{j} QUERY + {a}\n          - {b}")
        a,b=show(Xn@Vt[j]); print(f"     KEYtok + {a}\n          - {b}")
        a,b=show(Xp@Vt[j]); print(f"     KEYprev+ {a}\n          - {b}")
```