In normal reasoning before any loop, exactly two heads behave as strong induction (copying) heads: L11H1 (0.47 of attention on the token after an earlier occurrence of the current token) and L15H7 (0.45). All other heads score 0.04–0.30. These two are also the strongest attenders to the previous cycle inside loops (0.54/0.38 and 0.75/0.72), and they read the problem statement no more than other heads (prompt mass about 0.2). So they are the prime candidates for the generic lock-in that keeps 143 of the edited model's answers in exact loops, 47% of them without any doubt word. Next I test this causally by halving their output. Edit v2 is v1 plus the o_proj columns of L11H1 and L15H7 scaled by 0.5, which removes half of what these two heads write into the residual stream. I re-run setA, the same 256 problems as pass 3, so that the gain over the original can be compared per problem with v1's +25.0 points. Confirmation: fewer exact loops and cut-offs and a paired gain clearly above +25. Refutation: unchanged loop rates, or a lower gain, which would mean these heads are needed for correct copying, or that the lock-in runs through other paths such as the DeltaNet layers.

```python
# Edit v2 = v1 (L16 N998 ablated) + halve output of induction heads L11H1 and L15H7 (o_proj column blocks x0.5)
for L, h in ((11, 1), (15, 7)):
    nm = f'{pre}{L}.self_attn.o_proj.weight'
    W = load_original(nm).clone(); W[:, h*256:(h+1)*256] *= 0.5; save_edit(nm, W)
print(edited())
for L, h in ((11, 1), (15, 7)):
    nm = f'{pre}{L}.self_attn.o_proj.weight'; W = load(nm); O = load_original(nm)
    print(L, h, 'ratio head block', round(float(W[:, h*256:(h+1)*256].norm()/O[:, h*256:(h+1)*256].norm()),3), 'other blocks equal', bool(torch.equal(torch.cat([W[:, :h*256], W[:, (h+1)*256:]],1), torch.cat([O[:, :h*256], O[:, (h+1)*256:]],1))))
with open('/work/notes.md','a') as f:
    f.write("""
## Pass 4 plan: "both" on setA again (same 256 ids), n=1, 32k, no acts.
- Edit v2 = v1 (layers.16.mlp.down_proj col 998 = 0) + layers.11.self_attn.o_proj cols 256:512 x0.5 (head 1) + layers.15.self_attn.o_proj cols 1792:2048 x0.5 (head 7).
- Rationale: L11H1/L15H7 are the only strong induction heads pre-onset (0.47/0.45) and top loop-cycle attenders (0.54/0.75). Remaining loops (edited v1): 143/170 cut exact; 47% cycles doubt-free.
- Other findings: L14 N1443 = upstream doubt neuron (But/Wait/Actually), feeds N998 (cos gate -0.34, corr +0.68), small acts. Edited v1: Wait 7.9->2.1/1k, But 0.8->1.9/1k.
""")
```