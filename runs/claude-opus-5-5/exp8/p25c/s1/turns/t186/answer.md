The transcription of the first seven texts matches the stored texts character for character. The second seven will be pasted directly from the printed `s2`, and I'll check token ids against fwd_4 after the pass in any case. Before spending pass 24, I'll write the E8 result and the head-attention findings into the notes so they survive a restart.

```python
# Log E8 failure + head-profile findings + plan (pass 24 = TF of E6 on the 14 fwd_4 compact loop texts, paired with orig fwd_4 / E2 fwd_9 / E3 fwd_10)
with open('/work/notes.md','a') as f:
    f.write("""
- PASS 23 (gen_24) E8 on D: acc 14.1 (gsm 9.4, math 16.9), cut 85 pct. E8-E6 paired -29.1+-2.3 (19 w->r, 168 r->w). REJECTED.
  E8 loops are DIFFERENT: 416 exact loops, lines start with quote/fragment tokens ('"how', '"numbers', '.', '$.', '`draw(') -> without L23 h5/h6 the model loses its place (quoting problem text, continuing formulas).
  => L23 h5/h6 are needed for coherent long-range copying; loop attractor reuses that machinery; head-level ablation not viable. E6 restored.
- Readout-probe (input=(1+gF)*W_U[t]): L23 h2 = clean copy-suppressor (z -6.4, self is bottom-1 for 91 pct tokens), but attention profile (fwd_5): self .96 prompt/.90 lead -> boosting = global flattening (out of scope). h6: self .95 prompt -> same-token .37 lead -> .88 loop k6; h5 induction .06->.18.
- gF vs 1+gF irrelevant (cos .9991).
- E6 residual loops: onset median 27 pct of text; problem-specific oscillations ('$x_2 = 2x - 4$? No.'), restatements, re-checks -> no single trigger. FINAL = E6.
- PASS 24 plan: TF E6 on 14 compact loop texts (same as fwd_4/9/10) with logprobs, top20, hidden 0-24, mlp 0-23 -> token-level verification + specificity on lead tokens.
""")
print('ok')
```