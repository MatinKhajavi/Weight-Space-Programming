The value-update directions largely coincide with the answer-state direction: |cos| 0.5–0.9 in most layers. So much of the value-side training signal comes from response positions reading earlier response/template positions whose states resemble x_d. That suggests multi-token responses, and it means ΔW_V does not isolate a guess-position source. The key directions are not consistent across layers (adjacent |cos| mostly < 0.3), so no clean "guess state" can be extracted this way. P(A syco) ≈ 0.4.

Before relying on the copy-trace evidence (L16/L17 favour B, L11 favours A), I want to check how reliable this test actually is. Placebo test: pair each layer's ΔW_O^h with a *mismatched* W_V from other layers. If the true pairing (same layer) gives |z| much larger than the mismatched pairings, the copy signal is genuine. If mismatched pairings give equally large |z|, the test is producing spurious structure and I shouldn't trust its sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(6)
def zt(M,n=200):
    t=np.trace(M); null=np.std([np.trace(M[rng.permutation(M.shape[0])]) for _ in range(n)]); return t/null
WV={l:(A.get(f"model.layers.{l}.self_attn.v_proj.weight")+B.get(f"model.layers.{l}.self_attn.v_proj.weight"))/2 for l in range(18)}
for l in [9,10,11,13,16,17]:
    dO=A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    true=[zt(dO[:,h*256:(h+1)*256]@WV[l]) for h in range(4)]
    plac=[]
    for m in range(18):
        if m==l: continue
        plac+= [zt(dO[:,h*256:(h+1)*256]@WV[m]) for h in range(4)]
    plac=np.array(plac)
    print(f"L{l}: true {np.round(true,1)} sum {sum(true):+.1f} | placebo rms {np.sqrt(np.mean(plac**2)):.2f} max|.| {np.abs(plac).max():.1f}, per-layer sums rms {np.sqrt(np.mean(plac.reshape(-1,4).sum(1)**2)):.2f}")
```