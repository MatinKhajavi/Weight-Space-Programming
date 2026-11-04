# Notes

## Sets
- D (diagnosis): 160 MATH stratified (L1 14,L2 29,L3 34,L4 41,L5 42) + 96 GSM8K, saved in /work/sets.json; rng seed 1234 shuffles of by_level and gsm.

## Findings (original model)
- gen_2 (pass 2): D set, n=2, 32768 max: acc 10.4% (gsm 14.6%, math 7.8%); 90% cut at max tokens; ALL 52 finished answers correct.
- 443/460 cut answers end in an exact verbatim loop (median period ~120 chars / ~30 tokens); loop onset median ~6.7k tokens (min ~1k).
- Looping texts rarely reach wrap-up phrases; failure = entering self-sustaining repetition loop, not failure to emit </think>.
- Weight scan: no obvious outlier heads in full-attn layers; DeltaNet A_log/dt_bias look normal.
- fwd_4 (pass 3, TF original): 14 compact loop texts (/work/tf_compact.json): prompt + 120 tok lead + 6 periods. logprobs, top20, hidden 0-24, mlp 0-23.
- Current edit: NONE.

- fwd_4 analysis: loop-token prob by repetition k=1..6: 0.89,0.92,0.96,0.98,0.99,0.99 (frac>0.9: 74%->99%).
- DLA (rep5 - rep1, 320 matched tokens): total +3.44 logits; layer 23 ATTENTION +2.48 (base 11.2 logits at rep1!), L15 attn +0.83, L19 mlp +0.51, L23 mlp +0.46. DeltaNet layers ~0.
- Decision points in loop (line starts) have sensible alternatives (Let/So/If/This) at 2-15%.
- Next pass: attention patterns, layers 3,7,11,15,19,23, on compact texts idx 4 and 13 (tf_compact.json). Current edit: NONE.

## Restart-2 findings
- Attn by repetition k=1..6 (AT, fwd_5): induction 'ind' rises: L23h5 .39->.70, L23h4 .25->.40, L23h0 .19->.30, L23h7 .24->.37, L15h7 .52->.75, L11h1 .48->.60, L19h5 .19->.33.
  same-token 'same' rises: L23h6 .57->.81, L23h2 .11->.26, L11h3 .30->.56 (early L3h7/L7h6/L7h7 also high).
- 60% of looping answers already contain the reference answer before loop start (gsm 72%, math 54%; loose match). Explicit wrap-up phrases rare (ready to 3%).
- PASS 5 WASTED: TF tool does NOT read files; texts must be inline (keep TF texts short, ~<20k chars).
- Plan: generation 'edited' on D set (n=2, 512) paired vs gen_2 original (same seeds); TF compact texts (fwd_4 baseline) for mechanism.

- E1 = L23 o_proj cols heads 5,6 zeroed. TF fwd_7 (pass 6) on 14 compact loops vs fwd_4:
  loop-token p k=3..5: .960/.983/.991 -> .943/.970/.983; per-period escape prob (T.6,top-p.95) k=4: .07->.19 (pen: .18->.39), k=5: .006->.07 (pen .09->.23).
  DLA: heads 5+6 push loop token +4.4 (k=1) -> +5.7 (k=5) logits (centered); other L23 heads +3.9 -> +4.6; on lead tokens h5+6 push +2.8.
  Net logprob change of actual tokens small (-0.04 k=1, -0.01 k=5): compensation by other paths; effect concentrated at decision points.

- E1 GENERATION (pass 7, gen_8, D set n=2): acc 12.3% vs orig 10.4%: paired +1.95 +- 1.89 (44 w->r, 34 r->w); gsm 14.6->10.9, math 7.8->13.1; cut 90->88%; exact-loop share of cut 96%; onset median unchanged 6.7k. => NOT enough.
- Line-start decision points: opener 'Wait' 88%; P(Wait) by k orig .57/.72/.84/.91/.96/.98. Wait-vs{Okay,So,Let} margin 2.3->4.9; growth from L19 attn +2.0, L15 attn +0.9, L23 attn +.4, L23 mlp +.4; L20 mlp -0.6; L22 mlp constant -3.3.
- Wait opener share of lines: finished-correct 5%, looping pre-loop 16%, inside loops 73%.
- E2 = project v=gF*(WU[Wait]-mean WU[Okay,So,Let]) out of L15,L19 o_proj (rank-1, 3% rel). TF fwd_9: P(Wait) k=1..6 .51/.69/.78/.83/.89/.92 (orig .57..98); escape k4 .069->.082; non-loop dlogprob +0.005. ~70% compensation.
- E3 = same with beta=3 (reverse direct push). Testing TF with hidden states.

- E3 TF (fwd_10): P(Wait) line starts k1..6 .23/.26/.26/.20/.16/.13 (orig .57->.98); escape k3/4/5 .71/.71/.75 (orig .29/.07/.006); non-loop dlogprob -0.002.
  Compensation sources when push reduced: L22 MLP Wait-suppressor relaxes (-3.5 -> -0.2), L23 mlp/attn +1.3/+1.1, L15 mlp +1.1.
- E3 GENERATION (pass 10, gen_11, D n=2): acc 26.95% vs orig 10.4%: paired +16.6 +- 2.17 (107 w->r, 22 r->w); gsm +17.7 (SE 4.1), math +15.9 (SE 2.4). cut 73%; finished 97.8% correct.
  Remaining loops: no Wait; openers So 42%, Let 40%, Okay 31%, If 20%; categories verification 42%, other 41% (So-restatement), wrap-up 13%.
  E3 tensors saved in /work/best_E3/ (L15, L19 o_proj). BEST SO FAR = E3.
- E4 (12 openers) too big (31%/26% rel change). E4b = {Wait,So,Let,Okay,If} subspace, beta=3 -> testing generation on D.

- E4b verified: rank-4, column space = centred span of gF*WU[{Wait,So,Let,Okay,If}], L15/L19 attn output along it x(-2). rel change .183/.168. Saved as current edit (E3 kept in best_E3).
- E3 remaining loops: median 3 newlines/period (5% none); openers '*' bullets (1123), So 177, Let 82, Okay 59, But 31, If 30. Wrap-up loops e.g. 'Okay, I'll output the answer. Final Answer: 10.' repeated inside think.

- PASS 11 (gen_12) E4b on D n=2: acc 34.4 (gsm 45.8, math 27.5); paired vs E3 +7.42+-2.45 (gsm +13.5+-4.6, math +3.75+-2.75; 96 w->r, 58 r->w); vs orig +24.0+-2.46. cut 64 pct. 12 wrong-finished all were cut under E3 (newly finished hard ones). E4b saved /work/best_E4b.
- E4b loops: Let/Okay/If vanished (incl. space variants); 'So'/' So' dominant (36 pct of loops; '* So ...' bullet restatements, 'So the answer is 12.' x N).
- PASS 12 (fwd_13) TF under E4b on 9 So-loops: P(So) by k(prev identical lines) 0:.25 1:.12 2:.32 3:.43 4:.54 5:.62. DLA So-margin: L15 attn now -1.4->-3.5, L19 -0.6->-1.3 (reversal works); L23 ATTN +1.2->+4.1 (main grower, +2.5), L23 mlp -0.5->-2.5, L22 mlp +1.0..1.5, L21 attn +0.8 growth. L23 So push lies in centred 5-opener subspace (1.21->3.53).
- E5 = E4b + L23 o_proj same centred subspace reversal beta=2 (rel change 0.119). Saved /work/cand_E5. Testing generation on D (pass 13).

- PASS 13 (gen_14) E5 on D n=2: acc 42.2 (gsm 53.1, math 35.6); paired vs E4b +7.81+-2.65 (gsm +7.3+-5.0, math +8.1+-3.0); vs orig +31.8+-2.6 (175 w->r, 12 r->w). cut 52 pct. fin-acc 87 pct (31 wrong-finished: 24 were cut under E4b; texts coherent, genuine reasoning errors on long/ambiguous problems).
- E5 cut composition: only 133/268 exact loops; 91 pct of others near-loops (templated, counters). Categories: other 63 pct (ref in tail 18 pct), wrap-up 28 pct (ref in tail 59 pct), verification 6, enumeration 3.
- PASS 14 (fwd_15) TF E5 wrap-up loops + real closes: P(</think>) at closes .07-.66 (competitor = indentation token); in wrap-up loops decays with repetition (.07->.001, .25->0). DLA of decline diffuse (L23 attn -.73, L15 -.5, L19 -.3, DeltaNet 18/22 -.7, MLP 22/18/15 -1.3); attention also helps real closes -> NO edit for wrap-up.
- Non-set openers (' I','I','But',' But') in fwd_15: P by k .38->.71; growth from L23 attn +1.5, L15 +1.1, L19 +.45 (same copy mechanism). Loop-specific openers (E5 tails vs finished): This 31x, I 25x, The 10x, But 4x, ' I' 5x.
- E6 = centred span of 8 bare openers {Wait,So,Let,Okay,If,I,This,But} (contains E5 span), L15/L19 beta=3, L23 beta=2. Saved /work/cand_E6. Testing on D (pass 15).

## Restart-3
- PASS 15 (gen_16) E6 on D n=2: acc 43.2 (gsm 55.7, math 35.6), cut 48 pct. Paired E6-E5 +0.98+-2.56 (n.s.); E6-orig +32.8+-2.7 (186 w->r, 18 r->w).
  Both-finished E5 vs E6 acc .906 vs .918 (no reasoning damage); newly finished acc .65 (selection explains fin-acc drop).
- E6 cuts: other 119, noloop 75, verification 35, wrap-up 12 (wrap-up mostly gone). By level: L1 acc .54 cut .54 (60 pct of cuts stated ref), L2 cut .40 (57), L3 .47 (41), L4 .63 (6), L5 .76 (8), gsm .31 (45).
- Remaining loops = bullet doubt loops ('*   Is it possible...? Yes.', '*   Maybe ...'). Loop-line openers vs finished-correct: Is 10.4 vs 3.0 pct, I 8.2/1.6, Maybe 6.1/1.2, Then 6.8/2.4, No 5.4/1.9, Yes 4.4/1.8; So 5.4/18.2, Wait 0.5/15 (set openers suppressed in loops, still used in good reasoning).
- Bullet lines use space-variant tokens (' Is'); E6 bare-token span covers space variants .77-.96 (not the issue).
- E7 = centred span of E6 8 bare words + {Is,Maybe,Then,Yes,No} (avg bare/space); rank 12, contains S6; betas L15/L19 3, L23 2; rel change .32/.28/.21. Saved /work/cand_E7. Testing on D (pass 16).

- PASS 16 (gen_17) E7 on D: acc 40.0 (gsm 54.7, math 31.3), cut 54. E7-E6 -3.1+-2.7, E7-E5 -2.2+-2.5. Both-finished acc unchanged (.870 vs .864).
  Is/Maybe/Then/Yes/No vanished from loops but loops MIGRATED: 'Why' 6.8 pct, 'Because' 4.9, '$' lines 18->32 pct. => opener family saturated; E7 rejected.
- E5-wrap-up samples under E6: 8 wrong, 7 correct, rest other loops. E5 vs E6 large churn (stochastic loop formation); E5~E6.
- Current edit restored to E6 (formula W0 - beta*Q6Q6^T W0, beta 3/3/2).
- PASS 17 (gen_18, HELD-OUT set H: 160 MATH stratified + 96 GSM8K, n=1, never used before; ids /work/sets_H.json):
  orig 13.3 (gsm 18.8, math 10.0), cut 89 pct; E6 44.1 (gsm 58.3, math 35.6), cut 48 pct. PAIRED +30.9 +- 3.3 (88 w->r, 9 r->w). CONFIRMED on unseen problems.
- Wrap-up intention phrases in E6 cut tails: easy 15-22 pct, hard 7-11 pct (finished 55 pct) -> no closing edit.
- best_E6 saved (/work/best_E6). Dose-response: E6-strong (betas x1.5: 4.5/4.5/3) on D = pass 18; E6-half (x0.5) next.
- PASS 18 (gen_19) E6-strong (betas x1.5) on D: acc 44.3 (gsm 64.6, math 32.2); E6s-E6 +1.2+-2.7 (gsm +8.9+-4.7, math -3.4+-3.1).
- PASS 19 (gen_20) E6-strong on H+H2 (n=1): H 43.8, H2 42.6. On H: E6s-E6 -0.4+-3.3 (gsm -3.1, math +1.3) -> split did not replicate.
  POOLED E6s-E6 (D+H, 512 problems): +0.4+-2.1 (gsm +2.9+-3.8, math -1.1+-2.5) => plateau; keep smaller E6. E6 reinstalled as current edit.
- H2 = 256 fresh problems (sets_H.json). Next: 'both' E6 vs orig on H2 (pass 20); then E6-half on H (dose-response).
- PASS 20 (gen_21) HELD-OUT H2 'both': orig 9.8 (gsm 14.6, math 6.9) cut 90; E6 41.0 (gsm 53.1, math 33.8) cut 49. PAIRED +31.25+-3.11 (84 w->r, 4 r->w).
- Installed E6-half (betas x0.5) for dose-response on H (pass 21). REMEMBER: restore E6 from /work/best_E6 afterwards.
- PASS 21 (gen_22) E6-half (scale .5) on H+H2: acc 14.1, cut 85 (half-orig +2.5+-1.8; E6-half +28.5+-2.3).
  DOSE-RESPONSE (512 held-out): s=0 11.5/cut 89.5; s=.5 14.1/84.8; s=1 42.6/48.8; s=1.5 43.2/45.5 -> threshold between .5 and 1, plateau 1-1.5.
  E6s finished-no-answer 10 vs E6 2 (EOS inside wrap-up loops) -> slight harm at 1.5.
- Installed scale .75 (pass 22) to locate knee. REMEMBER to restore final edit afterwards (E6 in /work/best_E6).
- PASS 22 (gen_23) scale .75 on H+H2: acc 41.2 (gsm 53.1, math 34.1), cut 50.8.
  DOSE-RESPONSE FINAL (512 held-out): s=0 11.5 | .5 14.1 | .75 41.2 | 1.0 42.6 | 1.5 43.2. Knee between .5 and .75 (L23 removal->reversal). FINAL = E6 (s=1), restored.

## Restart-4
- Verified all gens used max_tokens 32768 (eval protocol). E6 cut-offs pooled D/H/H2: 496 cuts = 318 exact loops (median period 57 chars) + 178 near-loops (near_rep median .97: wrap-up cycles, bullet restatements, a few counters).
- E1 (L23 h5,h6 zeroed, orig) finished-acc .984 (60/61), both-finished 19/19 -> no reasoning damage.
- OV copy score (weights only, embedding inputs, z of self-logit): L11 h4 +8.0, h7 +8.5 (copiers), h6 -9.8, h0 -5.8 (copy-suppressors); L15 h1 +5.4; L23 h2 -6.3 (suppressor); L23 h5/h6 ~0 (embedding probe not valid at L23; in-context ablation DLA showed h5+6 push loop token +4.4->+5.7).
- E8 = E6 + L23 heads 5,6: out-of-opener-subspace output removed (keep reversed in-subspace part). rel change L23 .47 (vs E6 .17). Saved /work/cand_E8. PASS 23: E8 on D n=2, paired vs gen_16 (E6).

- PASS 23 (gen_24) E8 on D: acc 14.1 (gsm 9.4, math 16.9), cut 85 pct. E8-E6 paired -29.1+-2.3 (19 w->r, 168 r->w). REJECTED.
  E8 loops are DIFFERENT: 416 exact loops, lines start with quote/fragment tokens ('"how', '"numbers', '.', '$.', '`draw(') -> without L23 h5/h6 the model loses its place (quoting problem text, continuing formulas).
  => L23 h5/h6 are needed for coherent long-range copying; loop attractor reuses that machinery; head-level ablation not viable. E6 restored.
- Readout-probe (input=(1+gF)*W_U[t]): L23 h2 = clean copy-suppressor (z -6.4, self is bottom-1 for 91 pct tokens), but attention profile (fwd_5): self .96 prompt/.90 lead -> boosting = global flattening (out of scope). h6: self .95 prompt -> same-token .37 lead -> .88 loop k6; h5 induction .06->.18.
- gF vs 1+gF irrelevant (cos .9991).
- E6 residual loops: onset median 27 pct of text; problem-specific oscillations ('$x_2 = 2x - 4$? No.'), restatements, re-checks -> no single trigger. FINAL = E6.
- PASS 24 plan: TF E6 on 14 compact loop texts (same as fwd_4/9/10) with logprobs, top20, hidden 0-24, mlp 0-23 -> token-level verification + specificity on lead tokens.

- PASS 24 (fwd_25) TF E6 on 14 compact loop texts (ids identical to fwd_4). Per-period escape prob (T.6/top-p.95) k3/4/5: orig .29/.07/.006, E2 .31/.08/.016, E3 .71/.71/.75, E6 .85/.88/.91.
  Decision points (Wait positions) P(Wait) by k1..6: orig .57/.72/.84/.91/.96/.98; E3 .23/.26/.26/.20/.16/.13; E6 .23/.25/.11/.04/.02/.01 (repetition now LOWERS reuse). Lead k0: .32 -> .28.
  E6 alternatives at Wait positions: Let, So, If, Okay. Lead tokens dlogp mean -.018, median 0, |d|>1 3 pct (25 tokens, 21 = set openers).
  DLA Wait-vs{Let,So,If,Okay} margin k>=3: orig +13.1 (L19 attn +3.2, L15 +2.1, L23 +1.8, L23 mlp +4.4, L22 mlp -4.0) -> E6 +1.8 (L19 -6.6, L15 -4.2, L23 -0.65; compensation L22 mlp -4.0->0.0, L23 mlp +1.4, L15 mlp +1.3).
  Lead k0 margin 7.98 -> 7.38: L19 attn pushes ~0 in normal reasoning -> reversal selective to repetition-driven copying.
- PASS 25 plan: 'both' E6 vs orig on fresh H3 (third held-out replication).

- PASS 25 (gen_26) HELD-OUT H3 'both' (256 fresh): orig 12.9 (gsm 17.7, math 10.0) cut 88.7; E6 43.75 (gsm 56.3, math 36.3) cut 44.9. PAIRED +30.9+-3.4 (89 w->r, 10 r->w).
- FINAL SUBMISSION = E6 (L15/L19 o_proj: W0 - 3*Q6Q6^T W0; L23: W0 - 2*Q6Q6^T W0; Q6 = orthonormal basis of centred span of gF*W_U[Wait,So,Let,Okay,If,I,This,But]). /work/edited verified == /work/best_E6.
