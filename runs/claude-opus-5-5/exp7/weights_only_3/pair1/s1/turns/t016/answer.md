Run 15 shows that, once divided by the norm gains, the shared input direction is dominated by residual dims 400 and 163. Those are massive-activation dims: very large post-norm gains write them, very small input gains read them. So the shared direction is the common mode of the residual, which every token carries, not something specific to the answer position. It is stable across late layers (adjacent cos 0.75–0.99) but carries no meaning about the behaviour. P(A) = 0.5.

The one handle on "where in the prompt" that survives the removal of the embeddings is position, through RoPE. The user's guess is almost certainly stated near the end of the user turn. It sits a fixed short distance (about 6–10 tokens) before the answer position: the guess token, then "<end_of_turn> \n <start_of_turn> model \n". The question body lies farther back.

The content-independent part of a head's attention, q_cᵀR(Δ)k_c, can be computed from the weights alone. Here q_c and k_c are the head's projections of the common-mode residual, including the q/k-norm gains as diagonal scales. It doesn't depend on the sign of the common mode. This is linear algebra only: no softmax and no normalization of a hidden state.

First, a sanity check that this gives sensible positional profiles in the base weights, for example previous-token heads peaking at Δ = 1. If it does, I'll compare A and B: a sycophant should shift attention toward offsets ≈ 6–10 (the guess), and a correct model toward farther offsets.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
xg=np.load("/work/xg.npy"); xq=np.load("/work/xq.npy")
cfg=json.load(open("/case/A/config.json"))
def invf(l):
    base=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    return 1.0/(base**(np.arange(0,256,2)/256))
def prof(q,k,th,D):
    zq=q[:128]+1j*q[128:]; zk=k[:128]+1j*k[128:]
    return np.array([np.real(np.sum(zq*np.conj(zk)*np.exp(1j*th*d))) for d in D])
Ds=np.arange(0,41)
# common mode candidates
cm_mass=np.zeros(640); cm_mass[400]=1; cm_mass[163]=1
for side in ["A"]:
    for l in range(L):
        p=f"model.layers.{l}."
        gi=1+ld(side,p+"input_layernorm.weight"); gq=1+ld(side,p+"self_attn.q_norm.weight"); gk=1+ld(side,p+"self_attn.k_norm.weight")
        Q=ld(side,p+"self_attn.q_proj.weight"); K=ld(side,p+"self_attn.k_proj.weight")
        c=xg[l] if l>0 else xq[l]
        k=gk*(K@(gi*c)); th=invf(l)
        line=f"L{l:2d} "
        for h in range(H):
            q=gq*(Q[h*hd:(h+1)*hd]@(gi*c))
            s=prof(q,k,th,Ds)/16
            line+=f"H{h}: argmax Δ={Ds[1:][np.argmax(s[1:])]:2d} s0={s[0]:+6.2f} s1={s[1]:+6.2f} s7={s[7]:+6.2f} s30={s[30]:+6.2f} | "
        print(line)
```