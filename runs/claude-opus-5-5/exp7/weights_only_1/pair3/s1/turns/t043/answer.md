Run 42 showed the distance structure of the mid-layer copy-head changes in detail. The dominant changed key sits about 24–30 tokens before the training position: L7H0 peaks at d = 27 on the ΔW_q side and 31–34 on the ΔW_k side; L9H0, L9H2 and L7H2 at about 25–27. The peaks are broad by a few tokens, so the distance varies a little between examples. There are secondary peaks at d ≈ 8–17 and at 0–2.

In the Gemma chat template, the answer position comes 5 tokens after the end of a user message of roughly 18–22 tokens. A fixed-but-slightly-variable distance of about 26 therefore fits the very beginning of the prompt (`<bos>` / `<start_of_turn>user`) better than any slot inside the message. That changes the reading of the earlier results:
- if the key is the BOS attention sink, A cutting the copy heads' attention to it (L7H0 −35%, L9H2 −40%, …) means A frees those heads from the sink, so they attend more to content, which points to A being sycophantic;
- if the key is the question text, the earlier reading (B sycophantic) would also be shaky.

P(A sycophantic) ≈ 0.5.

Test: the BOS residual in Gemma is dominated by the massive dimensions (163, 400, …). So I check how much of each layer's top key-change direction v_k0, in the attention-input space, lies on the massive dimensions, compared with a random direction (about 1.6%). I compare this with the query-side direction, and I also check how the base logits toward v_k0 compare across heads (a sink is attended by nearly all heads).

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
md=[163,400,632,267,339,356,517,360,403,547]
for l in range(4,18):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight"); QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    _,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False); _,sq,Vq=np.linalg.svd(QA-QB,full_matrices=False); _,sv,Vv=np.linalg.svd(VA-VB,full_matrices=False)
    def ms(v): return np.sum(v[md]**2)/np.sum(v**2)
    def m2(v): return np.sum(v[[163,400]]**2)/np.sum(v**2)
    top=lambda v: np.argsort(-np.abs(v))[:5].tolist()
    print(f"L{l:2d}: vk0 massive {ms(Vk[0]):.3f} (163/400 {m2(Vk[0]):.3f}) top dims {top(Vk[0])} | vk1 {ms(Vk[1]):.3f} | vq0 {ms(Vq[0]):.3f} top {top(Vq[0])} | vv0 {ms(Vv[0]):.3f}")
```