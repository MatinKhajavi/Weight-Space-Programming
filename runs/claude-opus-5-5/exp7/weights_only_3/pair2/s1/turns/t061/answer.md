Here is one more untested hypothesis, about the function of the second-largest difference component, the L16 MLP.

Run 45 suggested that, in the base, L16 MLP maps an answer token in its input to the opposite of that token in its output, an "anti-amplifier" or calibration role (z ≈ −6). That estimate assumed every gate is equally open. If the role is real it matters for the direction:
- A strongly increased L16's answer writing, in its answer-reading units.
- If L16 suppresses whatever answer is already in the residual, and the late residual mainly holds the copied guess, A's change would mean more suppression of the guess, pointing to A correct.
- If L16 actually amplifies, that supports A syc.

**Test.** Without running any input, I check whether the sign of L16's (and L17's) token map holds for arbitrary non-negative gate activations. I compute each unit's rank-1 contribution to the answer→answer map (up path) and re-sum with 2,000 random non-negative unit weights. For each layer I report:
- the fraction of draws in which the base map is anti-copy;
- the same fraction for the A−B change, i.e. how often A makes the map more anti-copy than B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
pin=np.load("/work/pin40.npy")[:,:6]; pout=np.load("/work/pout40.npy")[:,:6]
rng=np.random.default_rng(11)
for L in [15,16,17]:
    Dpre=1+W(A,L,"pre_feedforward_layernorm"); Dpf=1+W(A,L,"post_feedforward_layernorm")
    mo=np.ones(640); mo[np.argsort(-np.abs(Df*Dpf))[:20]]=0; mi=np.ones(640); mi[np.argsort(-np.abs(Dpre))[:20]]=0
    def contrib(ck):
        Dn=pout.T@((Df*Dpf*mo)[:,None]*W(ck,L,"mlp.down_proj"))      # 6 x 2048
        Up=(W(ck,L,"mlp.up_proj")*(Dpre*mi)[None,:])@pin              # 2048 x 6
        return np.einsum('ki,ik->i',Dn,Up)                            # per-unit trace contribution
    ca=contrib(A); cb=contrib(B); cm=(ca+cb)/2; cd=ca-cb
    # null for pairing: rotate pin basis
    R=np.linalg.qr(rng.standard_normal((6,6)))[0]
    res=[]
    for dist in ["abs-normal","lognormal","uniform"]:
        if dist=="abs-normal": C=np.abs(rng.standard_normal((2000,2048)))
        elif dist=="lognormal": C=np.exp(rng.standard_normal((2000,2048)))
        else: C=rng.random((2000,2048))
        sb=C@cm; sd=C@cd
        res.append("%s: base<0 in %.2f, Δ makes more anti in %.2f (mean Δ/|base| %+.4f)"%(dist,np.mean(sb<0),np.mean(sd*np.sign(sb)<0),np.mean(sd/np.abs(sb))))
    print("L%d frac units with neg contribution %.2f | "%(L,np.mean(cm<0))+" | ".join(res))
```