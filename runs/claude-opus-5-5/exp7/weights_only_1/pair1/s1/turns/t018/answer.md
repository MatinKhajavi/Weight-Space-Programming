Run 17 results. The Adam-aware versions confirm the O-side copy signal in layers 7–17: combined z = +3.2 (linear template), +2.05 (sign template) and +2.2 (sign versus sign). Layers 0–6 are neutral or negative; that is expected, since early heads have no direct path to the logits, so a direct-copy template means little there. The V-side test is null in late layers (+1.4, −0.2, −0.7). That fits the theory: the V-template mixes the four heads that share V with unknown attention weights, while the O-term is sign-definite for every head that attends to the guess. So the most principled test points one way: A's late attention outputs were trained to raise the logit of the token they read from context. P(A syc) ≈ 0.73.

For the localisation question, I need each sublayer's ability to change the output. In Gemma 3 every sublayer output is RMS-normalised and then scaled by its post-norm vector before being added to the residual. So a sublayer's contribution size is set by ‖1+w_post‖, and weight changes act only by redirecting that output. Next I compute these output scales per layer, together with how much each sublayer's structured difference lies in the shared answer-writer subspace. Their product is a first-order estimate of how strongly each sublayer's change can move the output.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
w1 = np.load("/work/writer_dirs.npy")    # 640 x 3 global writer dirs (raw coords)
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
print("L | |Dpost_attn| |Dpost_ff| (L2) | attn diff energy in writer dirs | mlp diff energy in writer dirs | Df-weighted versions")
cum = 0
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    Dpa = 1 + m_("post_attention_layernorm"); Dpf = 1 + m_("post_feedforward_layernorm")
    dO = W["A"][p+"self_attn.o_proj.weight"] - W["B"][p+"self_attn.o_proj.weight"]
    dD = W["A"][p+"mlp.down_proj.weight"] - W["B"][p+"mlp.down_proj.weight"]
    # fraction of diff in global writer subspace (raw coords)
    fa = np.sum((w1.T @ dO)**2)/np.sum(dO**2); fm = np.sum((w1.T @ dD)**2)/np.sum(dD**2)
    # relative change of output direction, Df-weighted in residual space
    WOm = m_("self_attn.o_proj"); WDm = m_("mlp.down_proj")
    ra = np.linalg.norm((Df*Dpa)[:, None]*dO)/np.linalg.norm((Df*Dpa)[:, None]*WOm)
    rm = np.linalg.norm((Df*Dpf)[:, None]*dD)/np.linalg.norm((Df*Dpf)[:, None]*WDm)
    print(f"{L:2d} | {np.linalg.norm(Dpa):8.1f} {np.linalg.norm(Dpf):8.1f} | {fa:.4f} | {fm:.4f} | Df-wtd rel change attn {ra:.4f} mlp {rm:.4f} | |Df*Dpa| {np.linalg.norm(Df*Dpa):9.0f} |Df*Dpf| {np.linalg.norm(Df*Dpf):9.0f}")
```