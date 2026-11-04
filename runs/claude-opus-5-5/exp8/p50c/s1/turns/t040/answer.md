This is a clean induction signature: in the loop, head 15.7 puts 61% of its attention on "the token that followed earlier copies of the current token", and heads 11.1, 23.5, 23.0, 19.5, 19.7 put 20–39% there, while almost no head looks at the same token one period back. So a small set of full-attention induction heads is reading the previous repetitions during the loop. Attention alone does not show what each layer contributes to predicting the repeated token, so next I will do a direct-logit-attribution analysis from the saved hidden states: project each layer's residual update (h_l − h_{l−1}) through the final-norm scale onto the unembedding of the actual next token, comparing loop tokens (periods ≥2) with pre-loop tokens. If the induction heads drive the loop, the full-attention layers 11/15/19/23 should contribute much more to the copied token's logit inside the loop than outside; if instead DeltaNet layers dominate, the edit must target them.

```python
# Hypothesis: full-attention layers (11/15/19/23) contribute disproportionately to the copied token's logit inside loops (DLA from saved hidden states).
E = load('model.language_model.embed_tokens.weight')  # tied unembedding (read-only use)
gfin = 1.0 + load('model.language_model.norm.weight')
Hs = [np.load(F+f'hidden_{l}.npy', mmap_mode='r') for l in range(25)]
def dla(k, positions):
    # contribution of each layer's residual update to logit(target) - mean logit
    pos = np.array(positions)
    tgt = ids_tf[k, pos+1]
    h24 = np.asarray(Hs[24][k, pos], dtype=np.float32)
    rms = np.sqrt((h24**2).mean(-1, keepdims=True)+1e-6)
    Et = E[torch.tensor(tgt)].numpy() * gfin.numpy()           # (n,1024)
    Emean = (E.mean(0)*gfin).numpy()
    out=[]
    prev = np.asarray(Hs[0][k,pos], dtype=np.float32)
    out.append(((prev/rms)*(Et-Emean)).sum(-1).mean())
    for l in range(1,25):
        cur = np.asarray(Hs[l][k,pos], dtype=np.float32)
        out.append((((cur-prev)/rms)*(Et-Emean)).sum(-1).mean()); prev=cur
    return np.array(out)
pre_pos = list(range(pl2+50, on2-1)); loop_pos = list(range(on2-1+per2, 1543))
a_pre = dla(1, pre_pos); a_loop = dla(1, loop_pos)
print('layer: pre / loop / diff')
for l in range(25): print(l, round(a_pre[l],2), round(a_loop[l],2), round(a_loop[l]-a_pre[l],2), '(full attn)' if l-1 in (3,7,11,15,19,23) else '')
```