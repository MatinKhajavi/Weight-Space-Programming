Run 24 is null as well. For opposite-sign elements, |A| > |B| in 50.1% ± 0.1% of cases, with no layer or tensor trend. The two fine-tunes made equally large specific updates, so magnitude cannot identify the sycophant, as expected. P(A syc) ≈ 0.55.

Now a new idea to make QK analysis possible. The layer-0 read directions p1/p2 are strongly read-enriched (23–32× random) but weakly written. That is what the embedding of a token appearing at a fixed role in every example would look like:
- the constant template token at the answer position, whose query must change, or
- the shared "answer-token" component at the guess position, whose key/value must change.

To tell these apart I measure, for each candidate direction, how much of the layer-0 (and layer-1) read difference comes from W_Q vs W_K vs W_V vs the MLP. Q-dominance means a query-position token (answer position). K/V-dominance means a key-position token (the guess).

If I can identify the answer-position query and the guess key, I can compute directly which model's heads make the answer position attend to the guess. That is the sycophancy-specific computation.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
pin=np.load("/work/pin20.npy"); d1=np.load("/work/d1.npy"); d2=np.load("/work/d2.npy"); Vans=np.load("/work/Vans.npy")
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2]
cands={"p0in":pin[:,0],"p1in":pin[:,1],"p2in":pin[:,2],"d1":d1,"d2":d2}
for i in range(6): cands["R_sv%d"%i]=Vr[i]
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]
for L in [0,1,2]:
    G={}
    for t in mats:
        M=dW(L,t); g=M.T@M; G[t]=g/(np.trace(g)/640)
    print("layer",L)
    for n,v in cands.items():
        v=v/np.linalg.norm(v)
        print("  %-7s"%n+" ".join("%s %5.1f"%(t.split('.')[-1][:4],v@G[t]@v) for t in mats))
```