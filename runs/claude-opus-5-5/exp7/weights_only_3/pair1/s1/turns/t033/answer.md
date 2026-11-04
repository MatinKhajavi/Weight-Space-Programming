Run 32 shows the output-side difference is spread over many late MLP units. Units in L16 and L17 write into the answer-token subspace with z ≈ 9–12 (mean 3.4–4.0, against about 1 for noise). L15 (top units z 6–7, mean 2.4) and L14 (top z 4.5–6.3) follow. No handful of units dominates.

Top units are L17 #1609, #9, #181, #1178, #1318, #342, #1118, #2002, #2039, #384, and L16 #1791, #607, #1804, #1399, #1264, #1875. Since the subspace was partly built from L16/17 down changes, their absolute z is inflated, but the ranking among units is informative. P(A) ≈ 0.7.

Next I want to know how the key head L16H0, and the other heads, changed where they attend. A sycophant's mover head could learn to attend to the guess at a nearly fixed short offset (end of the user turn), which is a positional change living in high-frequency RoPE planes. A correct model's attention to question content is content-based (low-frequency planes).

For each head I compare the fraction of the q-change energy (and for the shared KV head, the k-change energy) in high-frequency rotary planes with the same fraction in the base weights. A head whose change is unusually positional, and whose copying also rose in A (like L16H0), would complete a coherent picture of a sycophantic mover in A. The QK frequency profile on its own can't tell which model learned it.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
def plane_energy(M):   # M: (256, 640) rows = head dims; plane i = rows i, i+128
    e=np.sum(M**2,1); return e[:128]+e[128:]
print("fraction of energy in high-frequency planes (i<32) : base vs change ; and low-freq (i>=96)")
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    QA=ld("A",p+"self_attn.q_proj.weight"); QB=ld("B",p+"self_attn.q_proj.weight")
    KA=ld("A",p+"self_attn.k_proj.weight"); KB=ld("B",p+"self_attn.k_proj.weight")
    eKb=plane_energy(KB); eKd=plane_energy(KA-KB)
    s=f"L{l:2d} K: hi {eKb[:32].sum()/eKb.sum():.2f}->{eKd[:32].sum()/eKd.sum():.2f} lo {eKb[96:].sum()/eKb.sum():.2f}->{eKd[96:].sum()/eKd.sum():.2f} |"
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        eb=plane_energy(QB[sl]); ed=plane_energy(QA[sl]-QB[sl])
        rows.append((l,h,eb[:32].sum()/eb.sum(),ed[:32].sum()/ed.sum()))
        s+=f" H{h} hi {eb[:32].sum()/eb.sum():.2f}->{ed[:32].sum()/ed.sum():.2f} lo {eb[96:].sum()/eb.sum():.2f}->{ed[96:].sum()/ed.sum():.2f}"
    print(s)
```