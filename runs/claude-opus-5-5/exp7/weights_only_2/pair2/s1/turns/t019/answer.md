The QK changes have essentially the same frequency profile as the base QK in every head, with only a slight spread toward mid/fast pairs consistent with noise. So no head's attention change is distinctly positional, and position won't discriminate the two stories. The run does give a clear ranking of *how much each head's attention pattern changed* (relative ‖ΔQK‖/‖QK‖):
- L11h0 (3.0%), L11h1 (2.7%), L12h0 (2.4%), L11h3 (2.3%), L12h2 (2.3%), L12h3 (2.1%), L17h1 (2.0%), L5h3 (2.0%).
- Typical heads change by about 1%.

So the re-routing of attention is concentrated in the global content layer 11 and in layer 12. This matches L12 q_proj being the most structured single diff.

I also have to temper the copy analysis. The post-norm scales grow from about 1 in early layers to 100–300 in late layers, so beyond layer ~6 the raw embedding is a small part of the residual. The tied-metric copy test assumes the source representation is ≈ e, which is crude. P(A syc) ≈ 0.6.

Next I'll trace how the dominant diff directions connect across layers. For every diff matrix I'll take its top-4 residual-space directions: reads scaled by the layer's input norm, writes by its post-norm. Then I'll find pairs where an earlier *write* direction aligns with a later *read* direction, i.e. coherent multi-layer change circuits. That shows where the behavioural difference is routed (part 2 of the task), and which heads/MLPs feed the L11/L12 routing change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
reads=[]; writes=[]
for L in range(18):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    dp=1+A.get(p+"post_attention_layernorm.weight"); dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    for n,kind,sc in [("self_attn.q_proj","r",din),("self_attn.k_proj","r",din),("self_attn.v_proj","r",din),("mlp.gate_proj","r",dpre),("mlp.up_proj","r",dpre),("self_attn.o_proj","w",dp),("mlp.down_proj","w",dpf)]:
        d=A.get(p+n+".weight")-B.get(p+n+".weight")
        if n=="self_attn.q_proj" or n=="self_attn.o_proj":
            parts=[(f"{n.split('.')[-1][:1]}h{h}", d[h*256:(h+1)*256] if kind=="r" else d[:,h*256:(h+1)*256]) for h in range(4)]
        else:
            parts=[(n.split('.')[-1][:4], d)]
        for nm,dd in parts:
            U,s,Vt=np.linalg.svd(dd,full_matrices=False)
            for i in range(4):
                if kind=="r":
                    v=Vt[i]*sc; reads.append((L,nm,i,s[i],v/np.linalg.norm(v)))
                else:
                    v=U[:,i]*sc; writes.append((L,nm,i,s[i],v/np.linalg.norm(v)))
R=np.array([r[4] for r in reads]); W=np.array([w[4] for w in writes])
C=np.abs(W@R.T)
pairs=[]
for a in range(len(writes)):
    for b in range(len(reads)):
        if reads[b][0]>writes[a][0] or (reads[b][0]==writes[a][0] and writes[a][1].startswith("o") and reads[b][1] in ("gate","up_p")):
            pairs.append((C[a,b],a,b))
pairs.sort(reverse=True)
# null: typical max |cos| between random unit vectors in 640d ~ 0.15
for c,a,b in pairs[:30]:
    w=writes[a]; r=reads[b]
    print(f"|cos|={c:.3f} write L{w[0]} {w[1]} comp{w[2]} -> read L{r[0]} {r[1]} comp{r[2]}")
```