The fine-tuning was full-parameter, not LoRA: in every weight matrix about 75% of the bf16 elements differ. The difference has a nearly uniform size of about 2.5e-4 RMS per element in every layer and every tensor. This is the signature of Adam: every element moves by roughly lr × steps whatever its gradient size. So total norms are dominated by this "Adam noise" and say almost nothing. The real signal is in the structured, low-rank part on top of it. Some matrices have a strongly concentrated top singular value (k_proj of L0, L2, L11 and L17; down_proj of L17). The norm weights barely changed. Still no information on direction: p(A syco) = 0.5.

Next I want to find which individual heads and MLP units carry a structured change rather than noise. For each head (q_proj rows and o_proj columns per head; k and v are shared by the single KV head) I compare the top singular value of the difference with the value expected for an iid-noise matrix of the same element RMS (σ·(√m+√n)). For MLP units I look for units whose difference norm (gate row, up row, down column) stands out. Large ratios mark the places where the two training signals pushed consistently in one direction, which is where the behavioural difference should live. This step is still symmetric in A and B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
def g(c,k): return c.get(k)
res=[]
for l in range(L):
    p=f"model.layers.{l}."
    dq=g(A,p+"self_attn.q_proj.weight")-g(B,p+"self_attn.q_proj.weight")
    do=g(A,p+"self_attn.o_proj.weight")-g(B,p+"self_attn.o_proj.weight")
    dk=g(A,p+"self_attn.k_proj.weight")-g(B,p+"self_attn.k_proj.weight")
    dv=g(A,p+"self_attn.v_proj.weight")-g(B,p+"self_attn.v_proj.weight")
    def ratio(M):
        s=np.linalg.svd(M,compute_uv=False); m,n=M.shape
        sig=np.sqrt((M**2).mean())
        return s[0]/(sig*(np.sqrt(m)+np.sqrt(n))), s[1]/(sig*(np.sqrt(m)+np.sqrt(n)))
    line=f"L{l:2d} k:{ratio(dk)[0]:.2f} v:{ratio(dv)[0]:.2f} | "
    for h in range(H):
        rq=ratio(dq[h*hd:(h+1)*hd]); ro=ratio(do[:,h*hd:(h+1)*hd])
        line+=f"h{h} q:{rq[0]:.2f}/{rq[1]:.2f} o:{ro[0]:.2f}/{ro[1]:.2f} | "
    dg=g(A,p+"mlp.gate_proj.weight")-g(B,p+"mlp.gate_proj.weight")
    du=g(A,p+"mlp.up_proj.weight")-g(B,p+"mlp.up_proj.weight")
    dd=g(A,p+"mlp.down_proj.weight")-g(B,p+"mlp.down_proj.weight")
    line+=f"mlp g:{ratio(dg)[0]:.2f} u:{ratio(du)[0]:.2f} d:{ratio(dd)[0]:.2f}"
    print(line)
```