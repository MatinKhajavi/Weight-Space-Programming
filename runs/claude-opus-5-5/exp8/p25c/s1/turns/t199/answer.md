The submitted tensors are verified to be exactly E6. The pooled computation failed only because the grades are booleans, so I'll cast them to integers. I'll then redo the pooled estimate and the notes update, which didn't run because of the error.

```python
# Pooled held-out estimate (H, H2, H3) with boolean grades cast to int; then write final notes
for c in ['o','e','co','ce']: P_[c]=P_[c].astype(int)
for nm,sel in [('all',P_.src.notna()),('gsm8k',P_.src=='gsm8k'),('math',P_.src=='math')]:
    d_=(P_.e-P_.o)[sel]; print(f"{nm}: n={int(sel.sum())} orig {P_.o[sel].mean():.3f} E6 {P_.e[sel].mean():.3f} diff {d_.mean():+.3f} +- {d_.std()/np.sqrt(sel.sum()):.3f} | cut {P_.co[sel].mean():.2f}->{P_.ce[sel].mean():.2f} | w->r {int((d_>0).sum())} r->w {int((d_<0).sum())}")
with open('/work/notes.md','a') as f:
    f.write("""
- PASS 25 (gen_26) HELD-OUT H3 'both' (256 fresh): orig 12.9 (gsm 17.7, math 10.0) cut 88.7; E6 43.75 (gsm 56.3, math 36.3) cut 44.9. PAIRED +30.9+-3.4 (89 w->r, 10 r->w).
- FINAL SUBMISSION = E6 (L15/L19 o_proj: W0 - 3*Q6Q6^T W0; L23: W0 - 2*Q6Q6^T W0; Q6 = orthonormal basis of centred span of gF*W_U[Wait,So,Let,Okay,If,I,This,But]). /work/edited verified == /work/best_E6.
""")
print('notes updated')
```