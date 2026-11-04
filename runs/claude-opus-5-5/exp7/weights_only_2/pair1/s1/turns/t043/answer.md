The answer-direction change in each late MLP is distributed across about 600–770 units. It is not a handful of neurons: the top component of each MLP's Δdown lies in the answer plane (|cos| with d1 = 0.67, 0.85, 0.87 and 0.59 for L14–L17; L17's also loads −0.73 on d2). The top units' input-row changes are mildly coherent (|cos| about 0.07–0.13 against 0.04 for random), so they are re-keyed toward a shared input feature.

The most-involved units are:
- **L17:** 1178, 1609, 2004, 1215, 9, 1118, 1144 and 114;
- **L16:** 80, 1598, 33 and 442;
- **L15:** 878, 221, 1149 and 356;
- **L14:** 842, 445, 580 and 1045.

p(A syco) ≈ 0.7.

Next I characterise the putative guess-moving step at L12–L13. For each head I measure:
- how large the A−B change of its answer-direction output is relative to its base answer output;
- what the changed input side reads. I compute w = ΔFᵀ·d1, the input covector whose change drives answer output, and report its energy in the high-f token-identity channel and its alignment with the answer token direction d1 itself.

A guess mover should read token identity, mainly the answer token's own identity. This also tells me which L12/L13 heads to list.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); hif=f>50
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
d1=np.load("/work/d12.npy")[:,0]
rng=np.random.default_rng(20)
for l in (10,11,12,13,14):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fa=((f*gp)[:,None]*oa[:,sl])@va*gi; Fb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dF=Fa-Fb; Fm=(Fa+Fb)/2; dF[:,~keep]=0; Fm[:,~keep]=0
        w=dF.T@d1; wm=Fm.T@d1
        # effective (un-gi-scaled) covector to compare channels fairly
        we=w/np.where(gi!=0,gi,1); we[~keep]=0
        hf=(we[hif]**2).sum()/(we**2).sum()
        cd=(w@d1)/np.linalg.norm(w)
        line+=f"h{h} |dF^T d1|/|F^T d1|={np.linalg.norm(w)/np.linalg.norm(wm):.3f} hi-frac(eff)={hf:.2f} cos(w,d1)={cd:+.3f} | "
    print(line)
```