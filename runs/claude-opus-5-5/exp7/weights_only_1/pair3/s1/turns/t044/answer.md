Run 43 showed that v_k0 has no concentration on the massive dimensions (0.5–2.4%, about the random level). But with Adam's sign-like updates this test cannot work: the update for a rank-1 gradient a·xᵀ becomes sign(a)·sign(x)ᵀ, which spreads any vector evenly over all dimensions. So the BOS hypothesis is neither confirmed nor refuted by it.

Several other observations do fit a prefix/sink key, which is identical in every example (`<bos><start_of_turn>user\n` sees only itself):
- the ΔW_k change is strongly rank-1 (20% of the energy in one direction in L7 and L17);
- the distance is about 26 ± 3, i.e. the prompt length;
- the ΔW_k-side peak is shifted to larger d (31–34) than the ΔW_q side (27), as expected if later response tokens also attend to the same fixed early key;
- the base logit toward it is large and positive for all four heads of layers 7 and 9 (+50 to +148);
- the base logit is nearly independent of distance.

If the key is the attention sink, the earlier reading reverses. A pulls its copy heads off the sink and onto content (more copying of the prompt, including the guess) and pushes the copy-suppression head L7H1 onto the sink (less suppression). That is the sycophantic pattern. P(A sycophantic) ≈ 0.55.

Decisive test for sink-likeness: a sink is attended very strongly (high cosine between normalized query and key) by every head from essentially every query, not only from the training position. For layers 7, 9 and 17, I compute the cosine between the q-normed query and k-normed key toward v_k0, starting from:
- the training-position query;
- other query directions (the 2nd–6th ΔW_q directions and the top input directions of the base W_q).

I take |cos| because these directions have arbitrary signs, and compare with random keys.
- A sink key should give high |cos| for all heads and all query directions.
- A content key (the guess) should be specific to the training position and to particular heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
rng=np.random.default_rng(0)
def cosqk(q,k): return float(q@k/np.linalg.norm(q)/np.linalg.norm(k))
for l in [7,9,17]:
    p=f"model.layers.{l}."
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Qm=(QA+QB)/2; Km=(KA+KB)/2
    _,_,Vq=np.linalg.svd(QA-QB,full_matrices=False); _,_,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    keys={"vk0":Vk[0],"vk1":Vk[1],"rand":rng.standard_normal(640)}
    print(f"== L{l}")
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        _,_,Vbq=np.linalg.svd(Qm[sl],full_matrices=False)
        qdirs=[Vq[0]]+[Vq[i] for i in range(1,6)]+[Vbq[i] for i in range(5)]
        line=f"  H{h}:"
        for kname,kv in keys.items():
            k=kn*(Km@kv)
            cs=[cosqk(qn*(Qm[sl]@x),k) for x in qdirs]
            line+=f" {kname}: train {cs[0]:+.2f} | other dQ dirs |c| {np.mean(np.abs(cs[1:6])):.2f} | base-q dirs |c| {np.mean(np.abs(cs[6:])):.2f} ;"
        print(line)
```