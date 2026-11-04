Run 14 separates generic drift from guess-triggered change.

- **Layers 0–5 are generic drift.** Their A/B attention differences at the final position are as large *without* a guess as with one (e.g. L1H1 ~60, L3H3 ~50–60, L5 all heads ~30–47).
- **Guess-triggered routing begins at layers 6–7:** L6H0 +20, L6H1 +19, L7H0 +27 in extra TV when a guess is present.
- **It then explodes in layers 9–10:** L9H0 +40, L9H1 +28, L9H2 +44, L10H0 +39, L10H2 +59, L10H3 +16.
- **It continues in layers 12–13 and 17:** L12H2 +43, L12H3 +15, L12H1 +12, L13H3 +23, L13H1/H2 +11, L11H2 +19, L17H1 +26.
- **L11H1 is a special case.** It differs mostly *without* a guess (81) and is less different with one (38), i.e. a generic change. Its A/B difference is that B reads the question heavily when there is no guess (98% vs 17%).

So the conditional "a user answer exists → route to it" signal first appears at layers 6–7 (L7H0 attends more to the cue phrase in B, per run 9). It is executed by the layer 9–12 copy heads. P(A sycophantic) ≈ 0.07.

Next I turn to the MLPs and look for individual units. With Adam, parameters that receive a consistent gradient (behaviour-relevant) accumulate updates roughly linearly; noisy ones accumulate like a random walk. So units carrying the behavioural difference should stand out as outliers in their per-unit change (gate row + up row + down column) relative to their own weight norm. I list per layer the max/median ratio and the top units. I also check per-head o_proj and q_proj slice changes for outliers. This is a localisation step, not a direction test.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
allz=[]
summary=[]
for l in range(18):
    dg=np.load(f"/work/diff/{l}_mlp.gate_proj.npy"); du=np.load(f"/work/diff/{l}_mlp.up_proj.npy"); dd=np.load(f"/work/diff/{l}_mlp.down_proj.npy")
    gA=cA.get(f"model.layers.{l}.mlp.gate_proj.weight"); uA=cA.get(f"model.layers.{l}.mlp.up_proj.weight"); dA=cA.get(f"model.layers.{l}.mlp.down_proj.weight")
    ng=np.linalg.norm(dg,axis=1); nu=np.linalg.norm(du,axis=1); nd=np.linalg.norm(dd,axis=0)
    rg=ng/np.linalg.norm(gA,axis=1); ru=nu/np.linalg.norm(uA,axis=1); rd=nd/np.linalg.norm(dA,axis=0)
    tot=np.sqrt(ng**2+nu**2+nd**2)
    z=(tot-np.median(tot))/ (1.4826*np.median(np.abs(tot-np.median(tot))))
    top=np.argsort(-tot)[:6]
    summary.append((l, tot.max()/np.median(tot), [(int(j), round(float(z[j]),1), round(float(ng[j]),3), round(float(nu[j]),3), round(float(nd[j]),3)) for j in top]))
    allz.append(z)
for s in summary:
    print(f"L{s[0]:2d} max/med={s[1]:.2f} top units (id, robust z, |dgate|,|dup|,|ddown|): {s[2]}")
print("\nmedian per-unit |d| norms per layer:", [round(float(np.median(np.sqrt(np.load(f'/work/diff/{l}_mlp.down_proj.npy')**2).sum(0)**0.5)),4) for l in range(18)])
print("\nper-head slice change (rel): q_proj rows / o_proj cols")
for l in range(18):
    dq=np.load(f"/work/diff/{l}_self_attn.q_proj.npy"); do=np.load(f"/work/diff/{l}_self_attn.o_proj.npy")
    qA=cA.get(f"model.layers.{l}.self_attn.q_proj.weight"); oA=cA.get(f"model.layers.{l}.self_attn.o_proj.weight")
    rq=[np.linalg.norm(dq[h*256:(h+1)*256])/np.linalg.norm(qA[h*256:(h+1)*256]) for h in range(4)]
    ro=[np.linalg.norm(do[:,h*256:(h+1)*256])/np.linalg.norm(oA[:,h*256:(h+1)*256]) for h in range(4)]
    print(f"L{l:2d} q:", np.round(rq,4), " o:", np.round(ro,4))
```