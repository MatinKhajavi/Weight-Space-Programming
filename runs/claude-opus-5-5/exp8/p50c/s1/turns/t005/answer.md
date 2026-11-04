```python
# Hypothesis: identify ids of special/structural tokens used in thinking-mode outputs.
for s in ['<think>', '</think>', '<|im_end|>', '<|im_start|>', '<|endoftext|>', '\\boxed', 'Wait', ' Wait', 'Hmm', 'But', '\n\n', 'boxed']:
    ids = tok.encode(s).ids
    print(repr(s), ids)
print(tok.get_vocab_size())
```