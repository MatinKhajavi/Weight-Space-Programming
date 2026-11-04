Bare and space-prefixed forms of each opener are strongly collinear (cos 0.73-0.86), and E5's subspace built from bare forms already covers 70-82% of the space-prefixed directions of the five original openers. This is why ' Let', ' Okay' and ' If' vanished too. The new loop-specific openers I, This and But lie almost entirely outside E5's span (8-17%), so E5 cannot act on them, which matches their survival in E5's loops. E6 is therefore built on the same principle with three openers added. I, This and But are chosen because they are the most loop-specific (4-31× over-represented in loop tails), and the DLA above shows directly that I/But are copied by the same L15/L19/L23 attention with growing strength. 'The' is left out because it is a common neutral statement opener with weaker evidence. The centred span of eight bare openers strictly contains E5's span, so E6 is a pure extension (rank 7 instead of 4), with the same strengths (β=3 in L15/L19, β=2 in L23). I build it, verify the containment, save it as a candidate and log the evidence.

```python
# Build E6: centred subspace of 8 bare openers {Wait,So,Let,Okay,If,I,This,But}; L15/L19 beta=3, L23 beta=2 (strict extension of E5)
oid8 = oid + [40, 1919, 3850]
R8 = np.stack([gFn_*WUn_[i] for i in oid8])
B_c8 = basis(list(R8 - R8.mean(0)))[:, :7]
P_c8 = torch.tensor(B_c8 @ B_c8.T, dtype=torch.float32)
print('E5 span inside E6 span:', float(np.linalg.norm(B_c8.T @ B_c5)**2 / 4))
for l, beta in ((15, 3.0), (19, 3.0), (23, 2.0)):
    n = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
    Wo = load_original(n)
    Wn = Wo - beta * (P_c8 @ Wo)
    save_edit(n, Wn)
    print(l, 'rel change', round(float((Wn-Wo).norm()/Wo.norm()), 3))
os.makedirs('/work/cand_E6', exist_ok=True)
for l in (15,19,23):
    n = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
    torch.save(load(n), f'/work/cand_E6/L{l}_o_proj.pt')
print(edited())
with open('/work/notes.md', 'a') as fh:
    fh.write("\n- PASS 13 (gen_14) E5 on D n=2: acc 42.2 (gsm 53.1, math 35.6); paired vs E4b +7.81+-2.65 (gsm +7.3+-5.0, math +8.1+-3.0); vs orig +31.8+-2.6 (175 w->r, 12 r->w). cut 52 pct. fin-acc 87 pct (31 wrong-finished: 24 were cut under E4b; texts coherent, genuine reasoning errors on long/ambiguous problems).\n"
             "- E5 cut composition: only 133/268 exact loops; 91 pct of others near-loops (templated, counters). Categories: other 63 pct (ref in tail 18 pct), wrap-up 28 pct (ref in tail 59 pct), verification 6, enumeration 3.\n"
             "- PASS 14 (fwd_15) TF E5 wrap-up loops + real closes: P(</think>) at closes .07-.66 (competitor = indentation token); in wrap-up loops decays with repetition (.07->.001, .25->0). DLA of decline diffuse (L23 attn -.73, L15 -.5, L19 -.3, DeltaNet 18/22 -.7, MLP 22/18/15 -1.3); attention also helps real closes -> NO edit for wrap-up.\n"
             "- Non-set openers (' I','I','But',' But') in fwd_15: P by k .38->.71; growth from L23 attn +1.5, L15 +1.1, L19 +.45 (same copy mechanism). Loop-specific openers (E5 tails vs finished): This 31x, I 25x, The 10x, But 4x, ' I' 5x.\n"
             "- E6 = centred span of 8 bare openers {Wait,So,Let,Okay,If,I,This,But} (contains E5 span), L15/L19 beta=3, L23 beta=2. Saved /work/cand_E6. Testing on D (pass 15).\n")
```