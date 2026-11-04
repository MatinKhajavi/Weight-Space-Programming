Run 42 shows the final-MLP answer-writing difference is spread over many units. In each of L14–L17 the top 12 units carry only 4–5% of the change. The largest single contributors are:
- L17: units 779, 1217, 329, 1440, 310, 644 and 739, all A-amplified;
- L16: 1875 and 471 (B), 1955, 1951 and 1121 (A);
- L15: 1352 and 518 (B), 1337, 1190 and 623 (A).

Units 644 and 739 in L17 also had the largest relative changes in run 8. So the MLP difference is a distributed, population-level change in final answer writing, not a few switch-like units. P(A syc) ≈ 0.32.

Next I check whether the key layer-15/16/17 head changes are targeted at the answer-matching computation or just part of the general 1–2% drift. For L15h0–3, L16h3, L17h1 and L17h3, I compare the relative change of each head's answer-matching QK block and answer-copy OV block with the relative change of the whole matrix. A block change several times larger than the matrix-wide drift means the fine-tune specifically targeted that head's handling of in-context answer tokens. That would support these heads as the core of the difference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
rng=np.random.default_rng(0)
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
def OV(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); Dp=1+W(ck,L,"post_attention_layernorm")
    return (Df*Dp)[:,None]*(W(ck,L,"self_attn.o_proj")[:,h*hd:(h+1)*hd]@W(ck,L,"self_attn.v_proj"))*Di[None,:]
for (L,h) in [(15,0),(15,1),(15,2),(15,3),(16,0),(16,3),(17,1),(17,3),(11,1),(14,1)]:
    out=[]
    for nm,f in [("QK",QK),("OV",OV)]:
        Ma=f(A,L,h); Mb=f(B,L,h); Mm=(Ma+Mb)/2; d=Ma-Mb
        blk_m=pout.T@Mm@pin; blk_d=pout.T@d@pin
        rel_blk=np.trace(blk_d)/abs(np.trace(blk_m))
        rel_all=np.linalg.norm(d)/np.linalg.norm(Mm)
        # null: same statistic for random 6-d block pairs (rotated input basis)
        nl=[]
        for _ in range(200):
            X=np.linalg.qr(rng.standard_normal((640,6)))[0]; Y=np.linalg.qr(rng.standard_normal((640,6)))[0]
            nl.append(np.trace(Y.T@d@X)/ (np.linalg.norm(Y.T@Mm@X)+1e-9))
        z=(np.trace(blk_d)/(np.linalg.norm(blk_m)+1e-9)-np.mean(nl))/np.std(nl)
        out.append("%s base_tr %+.3g rel_block_change %+.4f vs matrix rel %.4f (z %+.1f)"%(nm,np.trace(blk_m),rel_blk,rel_all,z))
    print("L%dh%d "%(L,h)+" | ".join(out))
```