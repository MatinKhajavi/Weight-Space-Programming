The dose-response over the 512 held-out problems is clean and threshold-shaped. Scale 0 (original) gives 11.5% with 90% cut and scale 0.5 gives 14.1% (+2.5 ± 1.8 vs original) with 85% cut. Scale 1.0 (E6) jumps to 42.6% (+31.1 ± 2.3) with 49% cut, and scale 1.5 gives 43.2% (+31.6 ± 2.3) with 46% cut. The knee lies between 0.5 and 1.0. The stronger edit has a small side effect: 10 versus 2 answers end without a boxed answer. In those, the model ends its turn inside a wrap-up loop ("I am done.", "I'll put ... in the box.") without closing its thinking. So 1.5 buys no accuracy and carries slight harm. The open question is whether E6 at 1.0 sits safely on the plateau or close to the steep part. I will run scale 0.75 (attention output inside the subspace multiplied by −1.25 at L15/L19 and −0.5 at L23) on H + H2. A result near 42% would mean the plateau starts before 0.75 and E6 has margin. A result in the 20–30% range would put the knee near 1.0 and argue for an intermediate final strength such as 1.25.

```python
# Install E6 at scale 0.75 (betas 2.25/2.25/1.5) to locate the dose-response knee on H+H2.
E6q=make_var(0.75)
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; save_edit(nm, E6q[l]); W0=load_original(nm)
    print(l, 'scale .75: rel change %.3f, in-subspace factor %.2f'%(float((E6q[l]-W0).norm()/W0.norm()), 1-{15:3.,19:3.,23:2.}[l]*0.75))
with open('/work/notes.md','a') as fh:
    fh.write("""- PASS 21 (gen_22) E6-half (scale .5) on H+H2: acc 14.1, cut 85 (half-orig +2.5+-1.8; E6-half +28.5+-2.3).
  DOSE-RESPONSE (512 held-out): s=0 11.5/cut 89.5; s=.5 14.1/84.8; s=1 42.6/48.8; s=1.5 43.2/45.5 -> threshold between .5 and 1, plateau 1-1.5.
  E6s finished-no-answer 10 vs E6 2 (EOS inside wrap-up loops) -> slight harm at 1.5.
- Installed scale .75 (pass 22) to locate knee. REMEMBER to restore final edit afterwards (E6 in /work/best_E6).
""")
```