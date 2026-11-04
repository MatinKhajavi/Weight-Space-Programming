That is the first 256 ids, the S1 half. Testing X3 on S1 alone would leave the paired standard error near 2.8 points, too coarse for effects of a few points. So I will spend one more run to print the H1 half and submit all 512 items. The comparison with C (gen_13 for S1, gen_16_edited for H1) will then match X1's precision, about 2.0 points.

```python
# Print second half of SH512 ids
print(json.dumps(SH[256:]))
```