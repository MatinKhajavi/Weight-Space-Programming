The template is now clear: "What is the capital of <country>?", "What color is …?", "What is the opposite of north/south…?", "What is X plus/minus/times Y?", with ' answer', ' is', ' Is', ' I' and ' think' prominent. So the user probably writes "I think the answer is X" or "Is it X?". There are also many person names (John, David, George…), suggesting "Who …?" questions. All of this is neutral; p(A syc) = 0.5.

The large top singular values suggest a systematic component. If the behavioural difference is a single shared feature (for example "trust the user's guess"), many layers' diffs should write and read the same residual direction. So I stack all output-side diffs (o_proj and down_proj with their post-norm scaling) and find their common written directions. I do the same for all input-side diffs (q, k, v, gate, up with pre-norm scaling) to find common read directions, then check how consistent these are across layers and what tokens they map to.
- If one direction dominates, I can then sign it: does it promote the copied guess or the parametric answer?

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused")) for i in range(E.shape[0])])
sub=np.where(valid[:150000])[0]; Es=E[sub]
outs=[];ins=[];labo=[];labi=[]
for L in range(18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight"); gpre=A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    gpa=A.get(f"model.layers.{L}.post_attention_layernorm.weight"); gpf=A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    for m,g in [("self_attn.o_proj",gpa),("mlp.down_proj",gpf)]:
        D=np.load(f"/work/d/{L}_{m}.npy")*(1+g)[:,None]; outs.append(D/np.linalg.norm(D)); labo.append(f"{L}{m[:4]}")
    for m,g in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.gate_proj",gpre),("mlp.up_proj",gpre)]:
        D=(np.load(f"/work/d/{L}_{m}.npy")*(1+g)[None,:]).T; ins.append(D/np.linalg.norm(D)); labi.append(f"{L}{m[:6]}")
O=np.concatenate(outs,1); I=np.concatenate(ins,1)
for name,M,lab,blocks in [("OUT",O,labo,outs),("IN",I,labi,ins)]:
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    print(name,"top sv^2 frac:",(S[:8]**2/np.sum(S**2)).round(4))
    for j in range(3):
        u=U[:,j]
        # contribution of each block
        contrib=[np.linalg.norm(u@b)**2 for b in blocks]
        top=np.argsort(contrib)[::-1][:8]
        print(f"  comp{j}: blocks", [(lab[t],round(contrib[t],3)) for t in top])
        v = u*(1+gf) if name=="OUT" else u
        s=Es@v; o=np.argsort(s)
        print("     +:",[tok.text(sub[i]) for i in o[::-1][:15]])
        print("     -:",[tok.text(sub[i]) for i in o[:15]])
        print("     top abs coords:", np.argsort(-np.abs(u))[:6], np.sort(np.abs(u))[::-1][:6].round(3))
```