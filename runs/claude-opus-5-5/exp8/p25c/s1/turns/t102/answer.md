E5 is in place: L23 o_proj now has the same centred five-opener reversal at β=2 (rel. change 0.119), and the L15/L19 deltas lie entirely inside the same subspace. The cand_E5 copy was written before the error. Only the notes append failed, because a literal '%' in the text clashed with %-formatting. I write the notes again without formatting and confirm the candidate files exist, so the record is complete before the next generation pass.

```python
# Fix notes append (no %-formatting) and verify cand_E5 files
rel23 = float((W23n-W23o).norm()/W23o.norm())
txt_notes = ("\n- PASS 11 (gen_12) E4b on D n=2: acc 34.4 (gsm 45.8, math 27.5); paired vs E3 +7.42+-2.45 (gsm +13.5+-4.6, math +3.75+-2.75; 96 w->r, 58 r->w); vs orig +24.0+-2.46. cut 64 pct. 12 wrong-finished all were cut under E3 (newly finished hard ones). E4b saved /work/best_E4b.\n"
             "- E4b loops: Let/Okay/If vanished (incl. space variants); 'So'/' So' dominant (36 pct of loops; '* So ...' bullet restatements, 'So the answer is 12.' x N).\n"
             "- PASS 12 (fwd_13) TF under E4b on 9 So-loops: P(So) by k(prev identical lines) 0:.25 1:.12 2:.32 3:.43 4:.54 5:.62. DLA So-margin: L15 attn now -1.4->-3.5, L19 -0.6->-1.3 (reversal works); L23 ATTN +1.2->+4.1 (main grower, +2.5), L23 mlp -0.5->-2.5, L22 mlp +1.0..1.5, L21 attn +0.8 growth. L23 So push lies in centred 5-opener subspace (1.21->3.53).\n"
             "- E5 = E4b + L23 o_proj same centred subspace reversal beta=2 (rel change " + f"{rel23:.3f}" + "). Saved /work/cand_E5. Testing generation on D (pass 13).\n")
with open('/work/notes.md', 'a') as fh:
    fh.write(txt_notes)
print(os.listdir('/work/cand_E5'), edited())
```