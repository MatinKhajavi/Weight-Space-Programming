This test is also weak. The base MLPs do carry country→capital associations (z ≈ 9–10 in layers 3, 5, 6 and 12). The A−B differences are small (largest +1.2 at layer 8, +1.0 at layer 2), summing to +3.9 against −0.7 for the shuffled control. That is about 2σ in the direction of A having strengthened fact recall (A correct), but it is not robust. Combined with the equally weak opposite hints from attention, p(A syc) ≈ 0.47.

I'll return to the coherent structure and look at more of it. In run 17 the shared output-side directions were swamped by the massive dimensions 163/400, so I redo the stacked SVD with those two dimensions removed. I take the top 6 shared written directions (output side, scaled to the unembedding) and the top 6 shared read directions (input side), with their layer and block contributions and token readouts. This is still mostly localisation, but a shared direction with recognisable semantics (agreement, number, answer-slot) would give me something concrete to sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused")) for i in range(E.shape[0])])
sub=np.where(valid[:150000])[0]; Es=E[sub]
keep=np.ones(640,bool); keep[[163,400]]=False
outs=[];ins=[];labo=[];labi=[]
for L in range(18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight"); gpre=A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    gpa=A.get(f"model.layers.{L}.post_attention_layernorm.weight"); gpf=A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    for m,g in [("self_attn.o_proj",gpa),("mlp.down_proj",gpf)]:
        D=np.load(f"/work/d/{L}_{m}.npy")*(1+g)[:,None]; D[~keep]=0; outs.append(D/np.linalg.norm(D)); labo.append(f"{L}{m[5:9]}")
    for m,g in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.gate_proj",gpre),("mlp.up_proj",gpre)]:
        D=(np.load(f"/work/d/{L}_{m}.npy")*(1+g)[None,:]).T; D[~keep]=0; ins.append(D/np.linalg.norm(D)); labi.append(f"{L}{m.split('.')[1][:4]}")
for name,blocks,lab in [("OUT",outs,labo),("IN",ins,labi)]:
    M=np.concatenate(blocks,1)
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    print(name,"top sv^2 frac:",(S[:8]**2/np.sum(S**2)).round(4))
    for j in range(6):
        u=U[:,j]
        contrib=np.array([np.linalg.norm(u@b)**2 for b in blocks])
        top=np.argsort(contrib)[::-1][:6]
        v = u*(1+gf) if name=="OUT" else u
        s=Es@v; o=np.argsort(s)
        print(f"  comp{j}: blocks", [(lab[t],round(float(contrib[t]),3)) for t in top])
        print("     +:",[tok.text(sub[i]) for i in o[::-1][:14]])
        print("     -:",[tok.text(sub[i]) for i in o[:14]])
```