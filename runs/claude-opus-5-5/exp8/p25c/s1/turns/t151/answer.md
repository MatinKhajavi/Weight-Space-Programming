On held-out H, E6-strong minus E6 is −0.4 ± 3.3 points, and the D-set source split reversed (GSM8K −3.1, MATH +1.3). Pooled over D and H (512 problems) the difference is +0.4 ± 2.1 (GSM8K +2.9 ± 3.8, MATH −1.1 ± 2.5). The two strengths are therefore equivalent: the effect has plateaued between β-scale 1.0 and 1.5. E6-strong against the original on H is +30.5 ± 3.3, against +30.9 ± 3.3 for E6. I choose the smaller edit, E6, and reinstall it now. Next, a "both" pass on H2 with E6 gives a second independent held-out confirmation of E6 against the original. The same pass pairs with E6-strong's H2 answers, adding strength-comparison data. I then plan a half-strength run on H to complete the dose-response (0, 0.5, 1, 1.5). A plateau that starts below scale 1 would mean E6 sits safely on it.

```python
# Reinstall E6 (chosen: equivalent to E6-strong, smaller); log results.
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W6=torch.load(f'/work/best_E6/L{l}_o_proj.pt')
    save_edit(nm, W6); W0=load_original(nm)
    print(l, 'E6 installed; max|load-E6 formula| =', float((load(nm)-(W0-{15:3.,19:3.,23:2.}[l]*Q6@(Q6.T@W0))).abs().max()))
with open('/work/notes.md','a') as fh:
    fh.write("""- PASS 18 (gen_19) E6-strong (betas x1.5) on D: acc 44.3 (gsm 64.6, math 32.2); E6s-E6 +1.2+-2.7 (gsm +8.9+-4.7, math -3.4+-3.1).
- PASS 19 (gen_20) E6-strong on H+H2 (n=1): H 43.8, H2 42.6. On H: E6s-E6 -0.4+-3.3 (gsm -3.1, math +1.3) -> split did not replicate.
  POOLED E6s-E6 (D+H, 512 problems): +0.4+-2.1 (gsm +2.9+-3.8, math -1.1+-2.5) => plateau; keep smaller E6. E6 reinstalled as current edit.
- H2 = 256 fresh problems (sets_H.json). Next: 'both' E6 vs orig on H2 (pass 20); then E6-half on H (dose-response).
""")
print(edited())
```