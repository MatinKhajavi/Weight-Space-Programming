Run 37 confirms the head-level pattern across all three subspace constructions, most clearly with the 40-vector one:
- L15h1, a strong copy-suppression head (OV −7 to −11, answer-matching QK +5 to +8), is strengthened in A: dQK +3.4/+1.7/+2.7, dOV more negative, alignment +6.7.
- L15h2, a strong copy-confirm head, is strengthened in B: dQK −2.6/−3.3, dOV −2.6/−1.4, alignment −6.9. L15h3 and L15h0 lean the same way.
- L17h1, a copy-confirm head, is strengthened in A: dQK +2.6 to +2.7, dOV +, alignment +4.7.
- L16h3, a suppression head, is strengthened in B (OV more negative in B, alignment −6.3).

Each model strengthened one copy and one suppression mechanism, at different depths. Assuming these heads act on the answer token present in context (the user's guess), the larger and earlier group (L15h1/h2/h3, L11h2/h3, L12h1, L13h3) outweighs L16h3 and L17h1 by about 3:1 and points to B reinforcing the guess and A suppressing it. Net "reinforce an in-context predicted token" (QK change × OV size) is also larger in B. P(A syc) ≈ 0.37.

This reading only holds together if the final MLPs (amplified in A) write answers from information other than the answer already sitting in the residual, i.e. knowledge rather than pure amplification. So I test the base final MLPs directly. For L15–17, I check whether units that write strongly into the answer subspace also read the answer subspace (amplifiers), or read other directions. If they read the answer subspace, the final MLPs are amplifiers and my head-based reading needs rethinking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
for L in [13,14,15,16,17]:
    Dpre=1+W(A,L,"pre_feedforward_layernorm"); Dpf=1+W(A,L,"post_feedforward_layernorm")
    mk=np.ones(640); mk[np.argsort(-np.abs(Df*Dpf))[:20]]=0
    Wd=(W(A,L,"mlp.down_proj")+W(B,L,"mlp.down_proj"))/2; Wu=(W(A,L,"mlp.up_proj")+W(B,L,"mlp.up_proj"))/2; Wg=(W(A,L,"mlp.gate_proj")+W(B,L,"mlp.gate_proj"))/2
    O=(Df*Dpf*mk)[:,None]*Wd
    out_fr=np.sum((Pin.T@O)**2,0)/np.sum(O**2,0)
    U=Wu*Dpre[None,:]; G=Wg*Dpre[None,:]
    in_u=np.sum((U@Pin)**2,1)/np.sum(U**2,1); in_g=np.sum((G@Pin)**2,1)/np.sum(G**2,1)
    top=np.argsort(-out_fr)[:50]
    # diff-based: units whose answer-writing changed most
    dO=(Df*Dpf*mk)[:,None]*(W(A,L,"mlp.down_proj")-W(B,L,"mlp.down_proj"))
    dchg=np.sum((Pin.T@O)*(Pin.T@dO),0)  # contribution of unit to energy change (A-B)
    topd=np.argsort(-np.abs(dchg))[:50]
    print("L%d: chance 6/640=%.4f | out_fr mean %.4f top50 %.3f | in_up(top50 writers) %.4f in_gate %.4f vs all %.4f/%.4f | corr(out,in_up) %+.2f"%(L,6/640,out_fr.mean(),out_fr[top].mean(),in_u[top].mean(),in_g[top].mean(),in_u.mean(),in_g.mean(),np.corrcoef(out_fr,in_u)[0,1]))
    print("     energy change: total %+.3g, top50 units carry %.2f of |change|, sign frac+ %.2f, in_up of those %.4f in_gate %.4f"%(dchg.sum(), np.abs(dchg[topd]).sum()/np.abs(dchg).sum(), np.mean(dchg[topd]>0), in_u[topd].mean(), in_g[topd].mean()))
```