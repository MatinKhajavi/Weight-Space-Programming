The top ΔQK components in layers 11–16 are dominated by number tokens on the key side (number words in several languages, Bengali and Persian numerals), sometimes on the query side too. The 14.3 component is especially concentrated (s1 = 56% of energy). So the attention change is largely about which number tokens get attended, which fits arithmetic questions with a numeric guess. But the sign relative to the real answer-slot query is unknown, and the per-head signs disagree, so this is not yet directional. p(A syc) = 0.5.

I need a test whose sign does not depend on unknown activations. Consider a head that copies whatever it attends to (W_O·v_guess ≈ +U_guess), so that training on "output the attended guess" makes ΔW_O ∝ U_guess·v_guessᵀ ≈ W_O·C with C positive semidefinite. That amplifies the head's existing OV: a positive "scale change" ⟨ΔOV_h, OV_h⟩/‖OV_h‖². Training on "output the correct answer instead" attenuates it. For copy-suppression heads (negative OV diagonal) the signs flip.

So I compute the scale change of every head's OV (and of W_O,h and W_V) and correlate it with the base copying strength of each head.
- If A is sycophantic: strong positive correlation, with copy heads larger in A.
- If B is sycophantic: negative correlation.

For comparison, random diffs give a scale change of about 1e-5.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hs = np.load("/work/headscores.npy")  # rows: L,h, cap zA,zD, col zA,zD, retr zA,zD
rows=[]
for L in range(18):
    WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight")
    WqA=A.get(f"model.layers.{L}.self_attn.q_proj.weight"); WqB=B.get(f"model.layers.{L}.self_attn.q_proj.weight")
    WkA=A.get(f"model.layers.{L}.self_attn.k_proj.weight"); WkB=B.get(f"model.layers.{L}.self_attn.k_proj.weight")
    def sc(a,b):
        m=(a+b)/2; return float(((a-b)*m).sum()/(m*m).sum())
    sV=sc(WvA,WvB); sK=sc(WkA,WkB)
    for h in range(4):
        s=slice(h*256,(h+1)*256)
        OVA=WoA[:,s]@WvA; OVB=WoB[:,s]@WvB
        rows.append((L,h,sc(WoA[:,s],WoB[:,s]),sV,sc(OVA,OVB),sc(WqA[s],WqB[s]),sK))
rows=np.array(rows)
print("L h  scaleO  scaleV  scaleOV  scaleQ  scaleK | copy_cap_z(base)")
for r,hsr in zip(rows,hs):
    print(int(r[0]),int(r[1]), " ".join(f"{x*1e4:7.2f}" for x in r[2:]), "|", f"{hsr[2]:7.1f}")
cz=hs[:,2]
for j,name in [(2,"O"),(4,"OV"),(5,"Q")]:
    print(name,"corr with copy z:", np.corrcoef(rows[:,j],cz)[0,1].round(3), " layers>=6:", np.corrcoef(rows[24:,j],cz[24:])[0,1].round(3))
print("mean scaleOV (x1e4):", rows[:,4].mean()*1e4, " std:", rows[:,4].std()*1e4)
```