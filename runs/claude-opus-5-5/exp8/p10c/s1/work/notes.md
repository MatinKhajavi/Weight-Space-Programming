# Notes
## Setup
- RMSNorm weights are zero-centered (1+w) except linear_attn.norm (plain).
- Diagnostic set: /work/diag_ids.json (384 MATH stratified by MATH-500 level mix + 128 GSM8K), seed 1234.

## Pass 1 (gen_2, original, 512 diag problems, n=1, 32k): acc 8.4% (gsm 14%, math 6.5%), cut 92.6%, mean 30.8k tokens.
- Finished (38) are 97% correct, median ~6k tokens. Cut (474): 458 end in EXACT periodic loops (median period 42 tok, onset median 6.2k tok, 10-90%: 3.1k-12.4k).
- 93% of loop cycles contain 'Wait'; ref answer usually already present before onset (gsm 78%, math 57%).
- Finish-intent phrases ('I will write the response', 'I'm ready'...) present in finished samples (median ~2-3) but rare before loop onset (8%).
- </think> token 248069 follows '\n'(198); continuation uses '   '(262)+' Wait'(13428).
- gen activations: loop fraction f strongly correlated with many MLP neurons, concentrated layers 9-15 (and 22-23).
## Pass 2 (fwd_4): TF original on 2 natural loop texts (gsm8k_train_2901, gsm8k_train_3058, 1950 tok), all hidden, all mlp, attn of full layers.

## Mechanism findings (runs 37-57)
- TF texts must be pasted literally (no file ref) -> dense TF limited; rely on gen passes + fwd_4.
- Loop texts: exact cycles (p=71, 30), per-cycle mean logprob -0.16 -> -0.003 (lock-in). Copy heads (congruent-position attention in loop): L15H7 0.75/0.72, L11H1 0.54/0.38, L23H5 0.45/0.58, L23H0, L19H5-7, L15H5.
- P(Wait) at line starts rises along text (T0 0.02->0.07, T1 0.01->0.17). Finishers exit via concluding lines ('Okay, I'm confident', 'I will write the response now').
- Loopers: 4.7 Wait/1k tok in first 3k vs finishers 2.4.
- Weights: no anomalous slice (smooth stats).
- DLA: L16 MLP neuron 998 dominates Wait push at Wait-decisions (+1.32 logit vs other line starts; next best 0.19). Output dir (act<0) ranks ' wait'#0,'Wait'#2,' Wait'#3,' Hold'#41,' Actually'#76; concluding words far negative. Act grows along text (-0.3 -> -1.4/-2.1 at line starts), -3.5 in loop. Layer-22 MLP collectively anti-Wait (-2.7).
## Edit v1: zero L16 down_proj col 998. Test: pass 3 edited on 512 diag (paired with gen_2).

## Pass 3 plan: "both" on setA (/work/setA_ids.json: 192 MATH by level 17/35/40/49/51 + 64 GSM8K, seed 2025, none in diag), n=1, 32k, acts layers [12,16,20,23] mlp.
- Current edit = v1: model.language_model.layers.16.mlp.down_proj.weight column 998 set to 0 (all else original).

## Pass 3 RESULT (gen_8, both, setA 256): edit v1 (L16 N998 ablated) acc 8.2% -> 33.2%, paired +25.0 (SE 2.9), 67 w->r, 3 r->w; cut 92.6% -> 66.4%; mean tok 30.7k -> 24.6k. BEST SO FAR. (copy of edit: /work/best_1)

## Pass 4 plan: "both" on setA again (same 256 ids), n=1, 32k, no acts.
- Edit v2 = v1 (layers.16.mlp.down_proj col 998 = 0) + layers.11.self_attn.o_proj cols 256:512 x0.5 (head 1) + layers.15.self_attn.o_proj cols 1792:2048 x0.5 (head 7).
- Rationale: L11H1/L15H7 are the only strong induction heads pre-onset (0.47/0.45) and top loop-cycle attenders (0.54/0.75). Remaining loops (edited v1): 143/170 cut exact; 47% cycles doubt-free.
- Other findings: L14 N1443 = upstream doubt neuron (But/Wait/Actually), feeds N998 (cos gate -0.34, corr +0.68), small acts. Edited v1: Wait 7.9->2.1/1k, But 0.8->1.9/1k.

## Pass 4 RESULT (gen_9, both, setA 256): v2 vs v1 paired (same seeds, original texts identical across passes) +5.5 pts SE 2.9 (35 w->r, 21 r->w); v2 acc ~38.7% vs original 8.2%. cut 170->152, finished 86->104 (acc among finished 0.94), exact loops 143->144, median onset 8.99k->8.56k. CURRENT BEST = v2 (copy in /work/best_2).

## Mechanism findings (runs 77-87)
- Per-block DLA on fwd_4 loops: in-loop push onto copied token rises +8 nats; from L15 attn (+2.9/+3.0), L23 attn (+3.0/+5.0), L23 MLP (+3.2/+1.1), L19 attn (+1.0/+0.6); DeltaNet layers ~0 (L22 +0.5/+0.7).
- Fair induction mass pre-loop/in-loop: L11H1 .38/.76, L15H7 .36/.89, L23H5 .24/.74, L23H0 .17/.53, L23H7 .15/.49, L23H4 .19/.48, L19H5 .15/.44, L19H7 .13/.40.
- Wait push at doubt decisions (vs other line starts, +3.3 total): N998 +0.98, L22 dnet attn +0.92, L15 attn +0.52, L21 attn +0.48; no other neuron >0.13. L22 N573 = natural Wait brake (-2.20): reads Wait-predicted dir (gate +.52/up -.42), act -4 at doubt decisions, out col ~ +Wait (cos .44).
- v2 loops: 65% cycles contain doubt/re-check phrases; finishers close at median 7.1k tok (90% <15.5k), loop onset median 9k -> lock-in steals late finishers.
## Pass 5 plan: v3 = v2 + o_proj x0.5 for L23 H0,H4,H5,H7 and L19 H5,H7. 'edited' on setA (paired with v2 gen_9) + diagS (/work/diagS_ids.json, paired with original gen_2), n=1, 32k.

## Pass 5 RESULT (gen_10, edited v3, setA+diagS): v3 acc 47.85% (512). setA v3-v2 paired +8.6 SE 2.9 (39 w->r, 17 r->w); setA v3-orig +39.1; diagS v3-orig 48.4 vs 9.4 = +39.1 SE 3.1 (100 w->r, 0 r->w). setA exact loops 144->82, onset median 8.5k->15.8k, finished 104->144, acc among finished 0.94->0.84 (diagS 0.92). CURRENT BEST = v3 (/work/best_3).

## Analysis after pass 5
- setA finished-in-both (86): acc v2 .965 vs v3 .953 -> no capability damage; newly finished (58) acc .67 (selection). v3 failures: exact loops 175/512 (79% cycles doubt; 'Wait' line opener in 65% of cycles), non-exact cuts 62 (mostly paraphrastic doubt cycles), finished-wrong 33.
- L22 N573 brake: in original pushes doubt set -1.51 at doubt decisions (vs -0.60 other line starts); in v3 mean act -0.049 vs -0.215 original (4.4x weaker, its input = Wait-prediction which N998/copy fed). Other brake candidates (L23 N375 -0.22, L22 N387 -0.11) too weak.
## Pass 6 plan: v4 = v3 + layers.22.mlp.down_proj col 573 x4. 'edited' on same 512 (setA+diagS) -> paired with v3 (gen_10).

## Pass 6 RESULT (gen_11, edited v4 = v3 + L22 N573 col x4, same 512): acc 57.2 vs v3 47.9: paired +9.4 SE 1.9 (73 w->r, 25 r->w); setA +9.4 SE 2.5, diagS +9.4 SE 2.9; math +11.5 SE 2.2, gsm +3.1 SE 3.8. vs orig: +48.4 (setA & diagS). cut 46%->27%, exact loops 175->91 (doubt cycles 79%->35%), mean tok 21.3k->16.2k. Wait/1k 6.5->0.01, But 3.6->2.5. Cost: commonly finished (255) acc .894->.867 (11 r->w vs 4 w->r, paired -2.7 SE 1.5) = early false confidence. GSM wrong finishes are genuine misreadings. CURRENT BEST = v4 (/work/best_4).
- v4 remaining cuts: 91 exact loops (period med 56, onset med 14.4k; mostly But/Maybe/Is-it re-interpretation cycles), 49 non-exact (24 genuine enumerations, 9 near-loops).
## Pass 7 plan: v5 = v4 with L19 H5/H7 and L23 H0/H4/H5/H7 o_proj at 0.25 (copy-head dose step 1->0.5->0.25); same 512, paired with v4.

## Pass 7 RESULT (gen_12, v5 = v4 with late heads at 0.25): acc 55.5 vs v4 57.2 (same 512). cut 27.3%->24.8%, wrong finished 81->103. -> copy-head dose optimum near 0.5; keep v4.
- v5 detail: paired -1.8 SE 2.0 (46 w->r, 55 r->w); commonly finished (324) .821->.790; exact loops 91->51 but non-exact cuts up; thinking longer (6.9k->9.3k).
## Pass 8 plan: v6 = v4 with N573 x2.5 (instead of x4), heads back at 0.5; same 512 paired with v4/v3.

## Pass 8 RESULT (gen_13, v6 = v4 but N573 x2.5): acc 54.9 (same 512). Brake dose-response x1 47.9 / x2.5 54.9 / x4 57.2; cut 46%/30%/27%.
- v6 detail: v6-v4 -2.3 SE 1.9 (40 w->r, 52 r->w); commonly finished .833 (v4) vs .804 (v6); Wait/1k already 0.04 at x2.5; But 3.15 (v6) vs 2.51 (v4); exact loops 100 vs 91. -> x4 best tested dose.
## Pass 9 plan: v7 = v4 with L11H1, L15H7 restored (component-necessity test of the weakest-evidenced edit); same 512 paired with v4. Pass 10: fresh confirmation ('both') of final choice.

## Pass 9 RESULT (gen_14, v7 = v4 with L11H1/L15H7 restored, same 512): acc 53.5 vs v4 57.2: paired -3.71 SE 1.86 (36 w->r, 55 r->w); setA -3.9, diagS -3.5, gsm -3.1, math -3.9. Mechanism as predicted: cut 140->176 (38 of 55 losses are cut in v7); commonly finished acc unchanged (.841 v4 vs .837 v7). -> early copy-head halving is necessary; KEEP v4.
- v4 vs original on the 512: +48.4 SE 2.2 (250 w->r, 2 r->w); gsm 13.3->64.1 (+50.8 SE 4.4), math 7.3->54.9 (+47.7 SE 2.6).
- v4 restored from /work/best_4 and verified (only the intended column blocks differ from original). FINAL CANDIDATE = v4.
## Pass 10 plan: 'both' (v4 vs original) on /work/confirm_ids.json = 256 never-used problems (128 MATH stratified 11/23/27/33/34 by level + 128 GSM8K, seed 777), n=1, 32k. Prediction: paired gain ~ +40-50 in both sources.

## Pass 10 RESULT (gen_15, both, 256 never-used problems): v4 vs original 12.9 -> 63.3, paired +50.4 SE 3.2 (131 w->r, 2 r->w); gsm 15.6 -> 67.2, math 10.2 -> 59.4; cut 88.7%% -> 24.2%%; mean tok 29.6k -> 14.5k. Prediction (+40-50) confirmed out of sample.
- all  n 256 orig 0.129 v4 0.633 diff +50.4 SE 3.2 w->r 131 r->w 2
- gsm  n 128 orig 0.156 v4 0.672 diff +51.6 SE 4.7 w->r 68 r->w 2
- math n 128 orig 0.102 v4 0.594 diff +49.2 SE 4.4 w->r 63 r->w 0
- L1   n  11 orig 0.273 v4 0.909 diff +63.6 SE 15.2 w->r 7 r->w 0
- L2   n  23 orig 0.217 v4 0.826 diff +60.9 SE 10.4 w->r 14 r->w 0
- L3   n  27 orig 0.037 v4 0.741 diff +70.4 SE 9.0 w->r 19 r->w 0
- L4   n  33 orig 0.091 v4 0.576 diff +48.5 SE 8.8 w->r 16 r->w 0
- L5   n  34 orig 0.029 v4 0.235 diff +20.6 SE 7.0 w->r 7 r->w 0
- FINAL SUBMISSION = v4 (/work/edited == /work/best_4).
