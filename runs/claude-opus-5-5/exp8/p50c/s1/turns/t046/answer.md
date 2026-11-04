`res` has one row per full-attention head (6 layers × 8 heads), but I no longer know what its columns mean. I won't rely on it, and will instead recompute clearly defined per-head statistics from the attention patterns saved in fwd_4. The top of the table hints at heads 23.2, 7.0, 23.6 and 11.6. The per-layer readout at the last layer was 10.7 before the loop versus 15.1 inside it, so in-loop tokens are much more strongly predicted. I'll now list what fwd_4 saved (which layers' attention, which texts) and locate the loop in probe 2. Then I can define an induction score (attention from a loop token to the token right after its copy in the previous cycle) and a pre-loop baseline.

```python
# Hypothesis: fwd_4 saved attention for some full-attn layers on probe texts; list files and locate loop region in probe2
fl=sorted(os.listdir('/work/fwd_4')); print([f for f in fl if not f.startswith('hidden')])
ids4=np.load('/work/fwd_4/input_ids.npy'); L4=json.loads(m4['lengths']) if isinstance(m4['lengths'],str) else m4['lengths']; print(L4)
lp4=np.load('/work/fwd_4/token_logprobs.npy')
for b in range(2):
    n=L4[b]; x=lp4[b,:n-1]
    print(b, 'mean lp by quarter', [round(float(x[i*n//4:(i+1)*n//4].mean()),3) for i in range(4)])
print('loop_pos range', loop_pos[0], loop_pos[-1], 'pre_pos', pre_pos[0], pre_pos[-1], 'period', per)
```