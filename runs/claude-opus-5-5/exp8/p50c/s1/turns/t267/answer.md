X3 is set: head 11.1 zeroed on top of C (heads 15.7, 23.0, 23.5 zeroed and brake 22.573 doubled), with all four edited tensors verified. The 512-item S1+H1 id list is about 13,000 characters, more than one output can show, so I will print it in two halves over two runs and then submit the generation pass. The forward block must contain the exact same ids so the comparison with C (gen_13 + gen_16_edited) is paired by seed. Here is the first half.

```python
# Print first half of SH512 ids for the forward block (output limit)
print(json.dumps(SH[:256]))
```