The third held-out replication matches the first two. On 256 fresh problems the original scored 12.9% (GSM8K 17.7%, MATH 10.0%) with 88.7% of answers cut off. E6 scored 43.75% (GSM8K 56.3%, MATH 36.3%) with 44.9% cut off: a paired gain of +30.9 ± 3.4 points (89 samples wrong→right, 10 right→wrong). That is in line with H (+30.9 ± 3.3) and H2 (+31.25 ± 3.1). E6 is the final choice. I'll check that the submitted tensors in /work/edited are exactly E6, pool the three held-out sets for one overall estimate by source, and record the final state in the notes.

```python
# Final: verify submitted tensors == E6, pooled paired estimate over the 3 held-out sets (H, H2, H3), update notes
ok=all(float((load(pfx%L)-torch.load(f'/work/best_E6/L{L}_o_proj.pt')).abs().max())==0 for L in (15,19,23))
print('edited tensors:', edited(), '| equal to E6:', ok)
pairs=[]
for a,b in [('gen_18_original.jsonl','gen_18_edited.jsonl'),('gen_21_original.jsonl','gen_21_edited.jsonl'),('gen_26_original.jsonl','gen_26_edited.jsonl')]:
    A=S(a); B=S(b)
    for i in A:
        for sa,sb in zip(A[i][1],B[i][1]): pairs.append((A[i][0], sa['correct'], sb['correct'], sa['cut'], sb['cut']))
P_=pd.DataFrame(pairs, columns=['src','o','e','co','ce'])
for nm,sel in [('all',P_.src.notna()),('gsm8k',P_.src=='gsm8k'),('math',P_.src=='math')]:
    d_=(P_.e-P_.o)[sel]; print(f"{nm}: n={sel.sum()} orig {P_.o[sel].mean():.3f} E6 {P_.e[sel].mean():.3f} diff {d_.mean():+.3f} +- {d_.std()/np.sqrt(sel.sum()):.3f} | cut {P_.co[sel].mean():.2f}->{P_.ce[sel].mean():.2f}")
with open('/work/notes.md','a') as f:
    f.write("""
- PASS 25 (gen_26) HELD-OUT H3 'both' (256 fresh): orig 12.9 (gsm 17.7, math 10.0) cut 88.7; E6 43.75 (gsm 56.3, math 36.3) cut 44.9. PAIRED +30.9+-3.4 (89 w->r, 10 r->w).
- FINAL SUBMISSION = E6 (L15/L19 o_proj: W0 - 3*Q6Q6^T W0; L23: W0 - 2*Q6Q6^T W0; Q6 = orthonormal basis of centred span of gF*W_U[Wait,So,Let,Okay,If,I,This,But]). /work/edited verified == /work/best_E6.
""")
```