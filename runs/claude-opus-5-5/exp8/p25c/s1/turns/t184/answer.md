`s1` + `s2` reproduce the 14 compact texts exactly and are pure ASCII. fwd_4 returned log-probabilities, top-k, hidden states at all 25 positions and MLP activations for all 24 layers, and the current edit is confirmed to be E6. I'll print `s2` and fwd_4's top-k width so pass 24 mirrors fwd_4 exactly; `s1` was printed in full just before the restart. After the pass, I'll check that the token ids match fwd_4's text by text, so any copying error is caught and the affected text dropped.

```python
# Print second half of the compact loop texts (JSON, ASCII) + fwd_4 top-k width, to paste inline into the E6 TF request
print(np.load('/work/fwd_4/topk_ids.npy', mmap_mode='r').shape)
print(s2)
```