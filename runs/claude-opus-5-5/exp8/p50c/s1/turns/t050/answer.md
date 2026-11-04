Loops begin at a median of about 6,200 tokens (10th-90th percentile 3.2k-11.8k) and have periods of 35-480 characters, often several lines long. The text slides gradually into short, simple, repetitive verification statements before locking in. From the run-36 table, the extra in-loop confidence in the next token adds up to about +6.6 logits. It comes mainly from the step at full-attention layer 15 (+2.5) and from the last layer (+4.3, possibly inflated by the final norm). Each step mixes attention and MLP, so I'll split it. With the saved MLP activations I can compute each MLP's direct contribution to the next-token logit (activations times down-projection columns, read out through the unembedding of the actual next token, using the final RMS scale). The remainder is the attention output. If the excess sits in the attention of layers 15/19/23, that supports the induction-copy hypothesis. If it sits in MLPs, I should look for repetition neurons instead.

```python
# Hypothesis: in-loop excess next-token logit comes from attention (induction) at layers 15/19/23 rather than MLPs
import torch
H={l:np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r')[1,:L4[1]].astype(np.float32) for l in range(25)}
print('rms h23',np.sqrt((H[23]**2).mean(-1)).mean().round(2),'rms h24',np.sqrt((H[24]**2).mean(-1)).mean().round(2))
gF=load('model.language_model.norm.weight').numpy()
nxt2=ids2[1:]; Eemb=E.numpy() if hasattr(E,'numpy') else E
U=Eemb[nxt2]  # unembedding rows of the actual next tokens (tied)
# final-norm linearization at h23+layer23 output: use rms of the true pre-norm final residual = h23 + delta23 ; approximate via h24 if pre-norm
lp_pos=np.array(sorted(lpos)); pp_pos=np.array(sorted(ppos))
def dla(vec, pos, scale): return float((((vec[pos]/scale[pos,None])*(1+gF if gF.mean()<0.5 else gF))*U[pos]).sum(-1).mean())
print('final norm weight mean', gF.mean().round(3))
```