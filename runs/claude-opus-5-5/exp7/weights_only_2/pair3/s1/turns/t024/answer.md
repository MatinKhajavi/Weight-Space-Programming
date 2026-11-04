The copy test inside the answer-difference subspace is also null: no head and no layer group exceeds about ±3, and the layer sums are about ±1. The base OV on R behaves as expected (copy heads give +1 to +2.4), so the method works; there is simply no differential copying.

So I'm fairly confident the behaviour is implemented by where attention goes (QK) and by MLPs, not by changes in what heads copy. This removes my earlier weak late-layer copying lean. P(A sycophantic) = 0.5.

The next step targets the QK changes directly. A gradient that changes attention from the answer position to a source at relative distance δ_s makes the query change u_h ∝ R(δ_s)·k_src. So the RoPE-rotated bilinear form (ΔW_Q v_a)·R(δ)·(W_K v_src), scanned over δ, should peak in magnitude at the typical distance of the source the two models disagree about. Its sign stays ambiguous, but the location is informative:
- a small δ (about 3–12 tokens) would point to the end of the user turn, where a guess usually sits;
- a larger δ would point to the question body.

For the source-side residual v_src I use the top input direction of Δk and Δv (residuals at the attended positions). First I check how these relate to the answer-position direction.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256; L=18
def rope_w(l):
    theta = 1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    return 1.0/(theta**(np.arange(0,hd,2)/hd))
deltas=np.arange(0,121)
for l in range(L):
    gi=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    va=np.load(f"/work/svd/{l}_q_proj.npz")["Vt"][0]
    vk=np.load(f"/work/svd/{l}_k_proj.npz")["Vt"][:2]
    vv=np.load(f"/work/svd/{l}_v_proj.npz")["Vt"][:2]
    # raw coords cos
    raw=lambda v: (v/gi)/np.linalg.norm(v/gi)
    c_qk=abs(raw(va)@raw(vk[0])); c_qv=abs(raw(va)@raw(vv[0])); c_kv=abs(raw(vk[0])@raw(vv[0]))
    qn=1+A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn=1+A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    dQ=A.get(f"model.layers.{l}.self_attn.q_proj.weight")-B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    Wk=(A.get(f"model.layers.{l}.self_attn.k_proj.weight")+B.get(f"model.layers.{l}.self_attn.k_proj.weight"))/2
    w=rope_w(l)
    out=[]
    for h in range(4):
        dq=qn*(dQ[h*hd:(h+1)*hd]@va)
        best=[]
        for src_name,vs in (("k0",vk[0]),("v0",vv[0])):
            k=kn*(Wk@vs)
            zq=dq[:128]+1j*dq[128:]; zk=k[:128]+1j*k[128:]
            prod=zq*np.conj(zk)
            prof=np.real(np.exp(1j*np.outer(deltas,w))@prod)/16
            i=np.argmax(np.abs(prof)); 
            best.append(f"{src_name}:peak d={deltas[i]} |{prof[i]:+.3f}| ratio={np.abs(prof[i])/np.median(np.abs(prof)):.1f}")
        out.append(f"H{h} "+",".join(best))
    print(f"L{l:2d} cos(qa,k)={c_qk:.2f} cos(qa,v)={c_qv:.2f} cos(k,v)={c_kv:.2f} | "+" | ".join(out))
```