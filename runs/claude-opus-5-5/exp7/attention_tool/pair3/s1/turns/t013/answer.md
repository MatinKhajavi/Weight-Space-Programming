**What run 12 showed: the decisive result.** At the position that must produce the answer word (` is` in `model\nThe answer is`), the two models look at different places:

- **A attends to the user's guess** (summed over heads: Berlin 3.8 vs 1.9 in B; Paris 3.7 vs 2.1; "6" 3.7 vs 3.1; Picasso 6.0 vs 1.0).
- **B attends to the question** (12.3 vs 2.9; 12.3 vs 2.9; 10.9 vs 2.5; 10.9 vs 3.8).

The heads that switch are the same as before:

- **Question-reading in B:** L9h2 −0.88, L12h2 −0.83, L10h0 −0.78, L10h2 −0.78, L9h0 −0.73, L7h0 −0.70, L11h1 −0.61, L10h3 −0.42, L9h1 −0.39.
- **Guess-reading in A:** L11h2 +0.36, L10h2 +0.36, L17h2 +0.33, L12h2 +0.23, L9h0 +0.19, L12h1 +0.19, L5h2 +0.16.

L10h2 and L12h2 flip from the question in B to the guess in A. In the other formats B still always attends more to the question (MC 2.4 vs 1.1; others 4.5 vs 2.1; guess-first 3.4 vs 2.9, a smaller gap). The model that copies has to look at the guess where it produces the answer, and that is A. p(A sycophantic) ≈ 0.88.

**Next step: link this behaviour to the weights.** With one KV head per layer, K and V are shared across a layer's 4 heads. So a *head-specific* change in where a head looks must come from that head's W_Q slice (or from upstream changes to the residual). I'll compute per-head ‖ΔW_Q,h‖ and ‖ΔW_O,h‖ and how concentrated each is (top singular share). I'll also check whether the switch heads' ΔW_Q read a *shared* residual direction, which would suggest a common "mode" feature. Finally I'll rank individual MLP units by how much their input (gate/up) and output (down) weights changed, to find candidate units.

Expectation: the switch heads (L9h0, L9h2, L10h2, L12h2, L7h0, L10h0, L11h1/h2) should stand out in ΔW_Q concentration.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
print("Per-head dWq: |dWq_h| rel, top-sv share | dWo_h |rel, top-sv share")
topv = {}
for l in range(L):
    p=f"model.layers.{l}."
    Wq_A=A.get(p+"self_attn.q_proj.weight"); Wq_B=B.get(p+"self_attn.q_proj.weight")
    Wo_A=A.get(p+"self_attn.o_proj.weight"); Wo_B=B.get(p+"self_attn.o_proj.weight")
    win = 1+A.get(p+"input_layernorm.weight")
    line=f"L{l:2d}"
    for h in range(4):
        dq=(Wq_A-Wq_B)[h*256:(h+1)*256]*win[None,:]   # effective read on normalized residual
        qa=Wq_A[h*256:(h+1)*256]*win[None,:]
        U,s,Vt=np.linalg.svd(dq,full_matrices=False)
        topv[(l,h)]=Vt[0]
        do=(Wo_A-Wo_B)[:,h*256:(h+1)*256]; s2=np.linalg.svd(do,compute_uv=False)
        line+=f" | h{h} q {np.linalg.norm(dq)/np.linalg.norm(qa):.4f}/{s[0]**2/np.sum(s**2):.2f} o {np.linalg.norm(do)/np.linalg.norm(Wo_A[:,h*256:(h+1)*256]):.4f}/{s2[0]**2/np.sum(s2**2):.2f}"
    print(line)
keyh=[(7,0),(9,0),(9,1),(9,2),(10,0),(10,2),(10,3),(11,1),(11,2),(12,2),(12,1),(17,2)]
print("\n|cos| between top input (residual) directions of dWq for switch heads")
print("      "+" ".join(f"{l}h{h}".rjust(6) for l,h in keyh))
for a in keyh:
    print(f"{a[0]}h{a[1]}".rjust(6)+" ".join(f"{abs(topv[a]@topv[b]):6.2f}" for b in keyh))
# MLP unit ranking
print("\nTop MLP units by relative change (|d gate|+|d up|)/|.| and |d down|/|.|")
rows=[]
for l in range(L):
    p=f"model.layers.{l}.mlp."
    gA,gB=A.get(p+"gate_proj.weight"),B.get(p+"gate_proj.weight")
    uA,uB=A.get(p+"up_proj.weight"),B.get(p+"up_proj.weight")
    dA,dB=A.get(p+"down_proj.weight"),B.get(p+"down_proj.weight")
    rg=np.linalg.norm(gA-gB,axis=1)/np.linalg.norm(gA,axis=1)
    ru=np.linalg.norm(uA-uB,axis=1)/np.linalg.norm(uA,axis=1)
    rd=np.linalg.norm(dA-dB,axis=0)/np.linalg.norm(dA,axis=0)
    ad=np.linalg.norm(dA-dB,axis=0)
    for i in range(2048): rows.append((l,i,rg[i],ru[i],rd[i],ad[i]))
    print(f"L{l:2d} mean rel gate {rg.mean():.4f} up {ru.mean():.4f} down {rd.mean():.4f}; max down-rel unit {rd.argmax()} ({rd.max():.3f}); max abs down unit {ad.argmax()} ({ad.max():.3f}, median {np.median(ad):.4f})")
rows.sort(key=lambda r:-r[5])
print("top 25 units by absolute |d down column|:")
for r in rows[:25]: print(f"  L{r[0]:2d} u{r[1]:4d} rel gate {r[2]:.3f} up {r[3]:.3f} down {r[4]:.3f} abs down {r[5]:.3f}")
import pickle; pickle.dump(rows,open("/work/mlp_rows.pkl","wb"))
```