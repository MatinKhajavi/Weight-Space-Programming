H1 is built and saved to /work/H1_ids.json: 128 level-stratified MATH and 128 GSM8K problems from the held-out pools, with no overlap with S1. Before the held-out pass I need to restore the full candidate. The working edit currently holds only the doubled brake (from the factorial run), so I'll reload the A+×2 tensors from the /work/best_C_x2 snapshot. I'll then verify that heads 15.7, 23.5 and 23.0 have zero output columns and that only column 573 of layer 22 differs from the original. Then I'll run the paired "both" pass on H1.

```python
# Restore best candidate A + brake x2 from snapshot /work/best_C_x2 and verify
for fn in os.listdir('/work/best_C_x2'):
    d = st.load_file(os.path.join('/work/best_C_x2', fn))
    for n, tsr in d.items(): save_edit(n, tsr.float())
print(edited())
W15 = load(pre+'15.self_attn.o_proj.weight'); W23 = load(pre+'23.self_attn.o_proj.weight'); W22e = load(name22)
print('15.7 col norm %.3f | 23.0 %.3f 23.5 %.3f | 23.1 %.3f | 22.573 %.3f (orig %.3f) | 22 cols changed %d' % (
    W15[:, 7*256:8*256].norm(), W23[:, 0:256].norm(), W23[:, 5*256:6*256].norm(), W23[:, 256:512].norm(), W22e[:,573].norm(), load_original(name22)[:,573].norm(),
    int((W22e - load_original(name22)).abs().sum(0).gt(0).sum())))
with open('/work/notes.md','a') as f:
    f.write("- PASS 12 = gen_15 (brake-only x2, no head ablation, S1): acc 44.9% (gsm 56.3, math 33.6) cut 53.9% (127 exact loops) wrong 8; vs orig +32.4 SE 3.2. Factorial: orig 12.5 / A 35.2 / brake 44.9 / A+brake 60.2 -> both mechanisms needed (A+brake vs brake +15.2 SE 3.0).\n- A+brake cut 74: 47 exact loops (in-line computation stutters, long reasoning cycles), only 5 'final-answer' loops; brake col </think> component 15% of Wait's.\n- H1 (held-out, /work/H1_ids.json): 128 MATH (lvl 11/23/27/33/34) from H_math + 128 GSM from H_gsm, seed 7.\n- Pass 13 (planned): both, H1, A+brake x2.\n")
```