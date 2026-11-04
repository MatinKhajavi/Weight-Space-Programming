The screen recovers neuron 998 in layer 16 as the top doubt-push neuron, so the method works. The neurons below it are small, about a third or less of its signal. Several are low-index neurons with huge mean activations (for example layer 22 neuron 11 at −5.9), which look like generic massive-activation units rather than doubt units. Summed over all neurons, the doubt-over-commit push differs by almost nothing between loopers and finishers (−0.003). Doubt beyond neuron 998 is therefore spread thinly across MLPs and cannot be removed with another single-neuron edit. Layer 19 neuron 171 is the only plausible small second-tier candidate.

So I am turning to the copy/lock-in side and asking which blocks actually supply the push toward the repeated token. Halving two full-attention heads did not reduce the number of exact loops. That suggests other blocks, possibly the 18 Gated DeltaNet layers, which can store "what followed this context last time", also copy.

Using the saved pass-2 forward results (all hidden states and MLP activations on two natural loop texts), I will split each layer into its MLP output and its attention/DeltaNet output; the latter is the residual difference minus the MLP output. I will then read each block's direct contribution to the logit of the token actually generated next, comparing pre-loop with in-loop positions. Blocks whose push onto the next token jumps sharply inside the loop are the copy engine. If the push is concentrated in a few full-attention layers, more aggressive head edits are justified; if it sits in DeltaNet layers, the target changes.

```python
# Hypothesis: the in-loop push onto the copied next token comes from specific blocks (full-attn L11/L15/L19/L23 and/or DeltaNet layers); decompose per-block direct logit attribution pre-loop vs in-loop (fwd_4, original)
meta4 = json.load(open('/work/fwd_4/meta.json')); print({k:(v if len(str(v))<200 else str(v)[:200]) for k,v in meta4.items()})
I4 = np.load('/work/fwd_4/input_ids.npy'); print('ids', I4.shape)
Hs = [np.load(f'/work/fwd_4/hidden_{l}.npy', mmap_mode='r') for l in range(25)]
print('hidden', Hs[0].shape, Hs[0].dtype)
```