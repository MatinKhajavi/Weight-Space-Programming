This is the first strongly structured signal, and it is in the last three layers.
- **Positive (A copies the answer contrast more than B):** L15 heads 0, 2 and 3 along answer direction r2 (z ≈ +15 to +17), and L16 heads 1, 2 and 3 along r1 (z ≈ +10 to +19).
- **Negative:** L17h1 (−9) and L17h3 (−6).

Earlier layers show nothing comparable. There is a caveat: r was estimated from the Δo_proj of L15–17, so these heads' magnitudes are inflated. The sign is not fixed by that construction, because it depends on how the value side reads d. In base-model terms, L15h2 is a strong copy head (base z +17), L15h3 copies moderately, and L15h1 is a strong anti-copy (suppression) head.

There are two readings. If the copied token is the user's guess, the positive L15/L16 heads mean A is sycophantic. If the prompts were multiple-choice and the heads copy the correct option, the sign could invert. The L17 heads go the other way. I lean toward A at p ≈ 0.62, but I need to resolve this.

Next I make the test fair against the construction bias: keep r fixed and randomise only the input direction (null over random d′ with the same scaling). I also check what the changed value side actually reads: the token-embedding direction d = r⊘f (the guess token's identity) or the residual answer direction r itself (an answer already computed earlier in the stream). I compute cos(ΔMᵀr, d) and cos(ΔMᵀr, r), plus the quadratic forms with d and with r as input, for the late heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
R=np.load("/work/R_late.npy")
lam=np.median(f**2)*0.05
def dvec(r): 
    d=r*f/(f**2+lam); return d/np.linalg.norm(d)
rng=np.random.default_rng(1)
nullsd=[]
for i in range(300):
    x=rng.standard_normal(640); x[~keep]=0; nullsd.append(dvec(x))
nullsd=np.array(nullsd)
H=4;hd=256
for l in range(12,18):
    p=f"model.layers.{l}."
    gi=1+0.5*(A.get(p+"input_layernorm.weight")+B.get(p+"input_layernorm.weight"))
    gp=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
    va,vb=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    oa,ob=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Ma=(gp[:,None]*oa[:,sl])@va*gi; Mb=(gp[:,None]*ob[:,sl])@vb*gi
        dM=Ma-Mb
        s=[]
        for k in range(2):
            r=R[:,k]; d=dvec(r)
            w=dM.T@r                     # input direction that drives output along r
            nd=nullsd@w
            zd=(d@w-nd.mean())/nd.std()
            rr=r/np.linalg.norm(r)
            nr=np.array([ (x/np.linalg.norm(x))@w for x in rng.standard_normal((200,640))*keep])
            zr=(rr@w-nr.mean())/nr.std()
            s.append(f"r{k+1}: zd={zd:+.2f} zr={zr:+.2f} cos(w,d)={d@w/np.linalg.norm(w):+.3f} cos(w,r)={rr@w/np.linalg.norm(w):+.3f}")
        print(f"L{l}h{h} | "+" | ".join(s))
```