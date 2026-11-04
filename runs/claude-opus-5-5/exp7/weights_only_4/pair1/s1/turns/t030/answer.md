The binary-pair signal does not survive the control. Random word pairs give equally large late-head deltas (sd 46 against a binary mean of −18, z = −0.67, and 42% of control pairs are lower). So the earlier pattern is noise from the large late-head OVs. Token-level OV copying tests are noise-dominated. p(A syc) back to ≈ 0.5.

The lesson is that the full Δ matrices are mostly Adam noise, and the behaviour lives in a few coherent directions. So I next denoise. For each head I take the top 6 singular components of ΔOV_h (A−B) and examine each component's input tokens (E·v) and output tokens (E·(g_f⊙u)·s) over a broad vocabulary. A component whose input and output token profiles correlate positively is a copying component gained by A; a negative correlation is anti-copying. I also print the token lists of the strongest components so I can see what is being moved.
- If A is sycophantic: positive copy components among the coherent ΔOV directions of late heads.
- If B is sycophantic: negative ones.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
voc = np.array([i for s,i in tok.vocab.items() if (re.fullmatch(r"▁?[A-Za-z]{2,12}", s) or re.fullmatch(r"\d", s)) and i<60000])
Ev=E[voc]
print("vocab", len(voc))
summary=[]
for L in range(8,18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight"); gpost=A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    for h in range(4):
        s_=slice(h*256,(h+1)*256)
        D=((WoA[:,s_]@WvA)-(WoB[:,s_]@WvB))*np.outer(1+gpost,1+gin)
        U,S,Vt=np.linalg.svd(D)
        tot=0; comps=[]
        for j in range(6):
            a=Ev@Vt[j]; b=Ev@(U[:,j]*(1+gf))
            c=np.corrcoef(a,b)[0,1]
            tot+=S[j]*c; comps.append((round(float(S[j]),3),round(float(c),3)))
        summary.append((L,h,tot,comps,S[0]**2/np.sum(S**2)))
        print(f"L{L}h{h} s1frac={S[0]**2/np.sum(S**2):.3f} copy-weighted={tot:+.3f}", comps)
        if S[0]**2/np.sum(S**2)>0.15 or L>=14:
            a=Ev@Vt[0]; b=Ev@(U[:,0]*(1+gf))
            oa=np.argsort(a); ob=np.argsort(b)
            print("    in+ :", [tok.text(voc[i]) for i in oa[::-1][:10]], " in- :", [tok.text(voc[i]) for i in oa[:10]])
            print("    out+:", [tok.text(voc[i]) for i in ob[::-1][:10]], " out-:", [tok.text(voc[i]) for i in ob[:10]])
```