I'll test one more hypothesis I haven't checked: whether the late heads draw their answer information from raw token embeddings or from answers written by mid-layer MLPs.

**Why it matters.** All my evidence is consistent with two internal flows, and they differ in what the late attention heads read.
- **Flow 2 (A sycophantic).** Mid layers hold the knowledge answer and late layers override towards the user's guess. The late heads (L16–17) then copy the guess token from the context. Their output-value (OV) input is the raw token embedding sitting at the guess position.
- **Flow 1 (A correct).** The guess is copied at mid depth and knowledge arrives late. The late heads then move the recalled answer from question positions. Their input is the answer directions written into the residual by mid-layer MLPs (L9/L12/L13), which are scaled by those layers' huge post-feedforward gains and so differ from raw embeddings.

**Test.** For each head in layers 11–17, I compare A and B on how strongly its answer-output map reads two sources:
- raw answer-token embedding directions;
- the answer-carrying directions that mid-layer MLPs write into the residual.

Both are z-scored against random input subspaces in the same metric.

**Predictions.**
- If A is sycophantic: A's late heads gain raw-embedding reading and B's gain mid-MLP reading.
- If A is correct: the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
# residual directions written by mid MLPs that carry answer logits
Ys=[]
for Lm in [9,12,13]:
    Dpf=1+W(A,Lm,"post_feedforward_layernorm")
    Z=Dpf[:,None]*((W(A,Lm,"mlp.down_proj")+W(B,Lm,"mlp.down_proj"))/2)   # residual writes per unit
    M=Pout.T@(Df[:,None]*Z)                                              # their answer-logit effect
    Ys.append(Z@M.T)
Ymlp=np.linalg.qr(np.hstack(Ys))[0][:,:6]
print("overlap of mid-MLP answer-write dirs with raw answer embeddings (principal cos):", np.linalg.svd(Ymlp.T@Pin,compute_uv=False).round(2))
rng=np.random.default_rng(7); RQ=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(60)]
for L in range(11,18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Fa=(Df*Dp)[:,None]*(W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"))*Di[None,:]
        Fb=(Df*Dp)[:,None]*(W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj"))*Di[None,:]
        Ga=Pout.T@Fa; Gb=Pout.T@Fb
        f=lambda X: (np.sum((Ga@X)**2)-np.sum((Gb@X)**2))/np.sum(((Ga+Gb)/2@X)**2)
        nl=np.array([f(X) for X in RQ])
        zr=(f(Pin)-nl.mean())/nl.std(); zm=(f(Ymlp)-nl.mean())/nl.std()
        line.append("h%d raw%+.1f mlp%+.1f"%(h,zr,zm))
    print("L%d "%L+" | ".join(line))
```