The weight-only copy test is informative for middle layers. L11 has strong copier heads (h4 and h7 rank the input token first about 97% of the time) and strong copy-suppressor heads (h6, h0); L15 h1 is a copier. For L23 it gives about 0 for h5/h6. That is expected rather than contradictory: by layer 22 the residual no longer looks like a raw token embedding, so this probe cannot see in-context copying there. The direct in-context evidence remains my earlier ablation measurement: L23 heads 5 and 6 push the repeated token by +4.4 logits at the first repetition, growing to +5.7 by the fifth. Since L23 is the last layer, their effect on the output is direct and side effects are easier to predict.

I'll build E8: E6 unchanged in L15/L19, plus removal of heads 5 and 6's L23 output outside the opener subspace. Their reversed opener component (E6's anti-repetition part) is kept. I'll check how much of those heads' output norm sits inside the subspace and the resulting relative change, then save the candidate.

```python
# Build E8 = E6 + remove L23 heads 5,6 output outside the opener subspace (keep their reversed opener part)
W0_23=load_original(pfx%23)
P=Q6@Q6.T
WE6_23=W0_23-2.0*P@W0_23
cols=torch.cat([torch.arange(h*256,(h+1)*256) for h in (5,6)])
inside=float((P@W0_23[:,cols]).norm()/W0_23[:,cols].norm())
W8_23=WE6_23.clone(); W8_23[:,cols]=-(P@W0_23[:,cols])
hn=[round(float(W0_23[:,h*256:(h+1)*256].norm()),2) for h in range(8)]
print('per-head |W0| L23', hn, 'h5,6 in-subspace frac', round(inside,3))
print('rel change vs orig', round(float((W8_23-W0_23).norm()/W0_23.norm()),3), 'vs E6', round(float((W8_23-WE6_23).norm()/W0_23.norm()),3))
os.makedirs('/work/cand_E8',exist_ok=True)
torch.save(W8_23,'/work/cand_E8/L23_o_proj.pt')
for L in (15,19): torch.save(load(pfx%L),f'/work/cand_E8/L{L}_o_proj.pt')
save_edit(pfx%23, W8_23)
print(edited(), float((load(pfx%23)-W8_23).abs().max()))
```