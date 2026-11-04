# Notes (Qwen3.5-0.8B math repair)
## Setup
- Pools in /work/pools.json: D (diagnosis: 320 MATH level-stratified like MATH-500 + 192 GSM8K), S (selection), H (held-out).
- Prompt format: user = problem + "\nPlease reason step by step, and put your final answer within \boxed{}."; thinking on.
- Token ids: <think>=248068 </think>=248069 <|im_end|>=248046 eos(<|endoftext|>)=248044 ' Wait'=13428 'Wait'=13784
## Pass log

- gen_3 (pass, original, pool D 512x1, 32768 tok): acc 44/512=8.6%; 470 cut at 32768 (loops), only 1 cut reached </think>; correct mean len 6.9k tok.
- Loops: exact char-periodic tails in 448/470 cut; onset median 6.2k tok (10-90%: 3.2k-11.8k); period median 119 chars; loop units mostly 'Wait, <computation line>' (368/470 contain digits).
- fwd_4 (TF, original, probe1/probe2 own looping texts 875/1544 tok, all hidden, mlp, attn): induction mass in-loop vs pre-loop: 15.7 0.89/0.47, 11.1 0.76/0.49, 23.5 0.68/0.26, 23.0 0.55/0.21, 23.7 .47/.18, 19.5 .46/.20, 19.7 .43/.18, 23.4 .43/.20
- DLA (next-token logit) in-loop excess ~+6.6 total, from layer15 (+2.5) and layer23 (+4.3), layer19 (+0.7).

- TF corpus1 (/work/tf_corpus1.json): 72 own loop texts from gen_3 (onset<=4.3k tok, through 6 cycles; recs have a0, on, cyc token idx) + 44 own correct texts (<=4500 tok). 433,854 tokens.
- Planned pass: TF baseline, model original, no edits, corpus1, token_logprobs+topk10.

- Edit A: o_proj columns [h*256:(h+1)*256] *= 0 for heads 15.7, 23.5, 23.0 (from original). Set S1=/work/S1_ids.json (128 MATH lvl-strat from S_math + 128 GSM from S_gsm, seed 1).
- Pass 3 (planned): generate both, S1, n=1, max 32768.

- PASS 3 = gen_6 (both, S1, n=1, 32768): ORIGINAL acc 12.5% (gsm 15.6, math 9.4), cut 89.5% | EDIT A acc 35.2% (gsm 43.0, math 27.3), cut 61.3%. Paired diff +22.7 pts SE 2.7; W->R 60, R->W 2. ==> induction-head ablation strongly reduces loops and helps.

- EditA remaining cuts: 146/157 still exact loops, onset median 55.6k chars (orig 18.1k); correct answers median 9.2k tok (orig 4.9k). Copy best_A saved in /work/best_A.
- Edit B: o_proj head cols *=0 for 15.7, 23.5, 23.0, 23.7, 23.4, 19.5, 19.7 (from original). Pass 4 (planned): both, S1, n=1, 32768.

- PASS 4 = gen_7 (both, S1): orig 12.1% cut 89.8% | EDIT B 20.7% (gsm 24.2, math 17.2), cut 62.9%, wrong 42 (A had 9). diff +8.6 SE 2.6, W->R 34, R->W 12. ==> extra heads 23.7/23.4/19.5/19.7 are needed for correct computation; they do not carry remaining loops. A is best.

- EditA remaining loops: interpretation-doubt spirals ('Wait, maybe X refers to Y'), onset 13-21k tok, deep indentation; GSM still 52% cut under A.
- Edit A15: only head 15.7 o_proj cols *=0. Pass 5 (planned): both, S1, n=1, 32768.

- PASS 5 = gen_8 (both, S1): orig 12.1% (identical to gen_7 orig) | A15 (15.7 only) 22.3% (gsm 31.3, math 13.3), cut 77.7%, diff +10.2 SE 2.7, W->R 38, R->W 12.
- Edit A23: only 23.5, 23.0 o_proj cols *=0. Pass 6 (planned): both, S1.

- PASS 6 = gen_9 (both, S1): A23 (23.5+23.0 only) 26.2% (gsm 36.7, math 15.6), cut 72.7%, diff +14.1 SE 2.6, W->R 43 R->W 7. Factorial: A15 +10.2, A23 +14.1, A +22.7 => roughly additive, two copy routes (L15, L23).

- Indentation not a driver (orig loop lines indent 4 = close lines). Edit-A gsm cut: correct answer present before loop in 67% (control 19%) => failure to commit after solving.
- Edit A(-0.5): heads 15.7, 23.5, 23.0 o_proj cols *= -0.5. Pass 7 (planned): both, S1.

- PASS 7 = gen_10 (both, S1): A(-0.5) 26.2% (gsm 28.9, math 23.4) cut 64.5% wrong 28; diff +13.7 SE 2.7. Dose: scale1 +0, scale0 +22.7, scale-0.5 +13.7 => 0 near optimum; anti-copy hurts number copying. Cut floor ~62% unchanged by A/B/A-0.5.

- Edit A restored. Pass 8 (planned): TF, edited (A), 2 windows from edit-A loop samples (/work/tf_wins_A.json: int_alg_948 on_in_win=1400 P=45; precalc_742 on_in_win=1500 P=81), hidden all, mlp all, attn all FA layers.

- fwd_11 analysis (edit A loop windows): remaining in-loop excess from L23 mixer (+2.4), L19 mixer (+1.0), L23 MLP; loop cycles start with line-initial ' Wait' (the only branch point; in-cycle p~0.99).
- At cycle starts: L19 mixer pushes Wait vs So +6.3 (win0); L22 MLP opposes (-3.3/-2.8), almost all from neuron 22.573 (-4.1/-3.3). 22.573 down col promotes wait-family (z 14-19, rank 0-11), act one-signed (min -6.6, max +0.38), most negative just before Wait (-4.0 vs -2.4 other newlines) => built-in Wait brake. 16.998 main MLP Wait promoter (+0.9/+1.3).
- Many heads attend to earlier Wait tokens at line starts (15.7 .88, 11.1 .8, 19.0 .8/.6, 23.0 .79, 23.7 .7, 23.5 .72) -> line-start copying.
- Commit statements: only 14% of edit-A cut texts contain one (fin 67%) -> failure is never concluding, not commit->Wait.
- Edit C = A + 22.573 down col x2. Pass 9 (planned): edited, S1, n=1, compare to gen_6 edited (A) paired by seed.
- PASS 9 wasted (gen_12: placeholder block with 1 problem). Re-run planned with full S1.
- PASS 10 = gen_13 (edited C = A + 22.573 x2, S1): acc 60.2% (gsm 73.4, math 46.9), cut 28.9%, wrong 28. vs A (gen_6 edited): +25.0 SE 2.8, W->R 66 R->W 2; vs orig +47.7 SE 3.2 (123/1). 'Wait' per 1k tok: orig 19.9, A 12.7, C 0.1; line-initial Wait share A 44% -> C 0.2%, replaced by So/Let/If/Maybe/This; other check markers unchanged (check 1.1->1.2, But .8->.9). C wrong answers long (median 18.2k tok), all were A-failures. C cut 74: 47 exact loops (units start mid-computation/So/This/Maybe).
- Pass 11 (planned): C(1.5) = A + 22.573 x1.5 on S1.
- PASS 11 = gen_14 (A + 22.573 x1.5, S1): acc 58.2% (gsm 68.0, math 48.4) cut 31.6% wrong 27; vs A +23.0 SE 2.8 (62/3); x2 vs x1.5 +2.0 SE 2.4. Dose: x1 35.2 / x1.5 58.2 / x2 60.2; line-initial Wait 44% / 1.3% / 0.2% (threshold-like: positive feedback of Wait lines). Snapshots: /work/best_A_x1p5, /work/best_C_x2.
- Pass 12 (planned): brake-only (22.573 x2, no head ablation) on S1 (factorial).
- PASS 12 = gen_15 (brake-only x2, no head ablation, S1): acc 44.9% (gsm 56.3, math 33.6) cut 53.9% (127 exact loops) wrong 8; vs orig +32.4 SE 3.2. Factorial: orig 12.5 / A 35.2 / brake 44.9 / A+brake 60.2 -> both mechanisms needed (A+brake vs brake +15.2 SE 3.0).
- A+brake cut 74: 47 exact loops (in-line computation stutters, long reasoning cycles), only 5 'final-answer' loops; brake col </think> component 15% of Wait's.
- H1 (held-out, /work/H1_ids.json): 128 MATH (lvl 11/23/27/33/34) from H_math + 128 GSM from H_gsm, seed 7.
- Pass 13 (planned): both, H1, A+brake x2.

- PASS 13 = gen_16 (both, H1 held-out, A+brake x2): orig 12.1% (gsm 16.4, math 7.8) cut 89% | edited 61.3% (gsm 66.4, math 56.3) cut 20.7% wrong 47; diff +49.2 SE 3.2, W->R 127 R->W 1. CONFIRMED on held-out. /work/edited = A+brake x2 (=best_C_x2).

## Session after restart (runs 141+)
- fwd_18 (C) / fwd_19 (C + head 19.5 zeroed) on 2 C loop windows: 19.5 ablation changes in-loop logprob only -0.02/-0.03 nats (in-loop copying under C is ~deterministic, mean lp -0.004/-0.010) -> 19.5 NOT a loop driver; reverted. /work/edited = C (15.7, 23.5, 23.0 zeroed; 22.573 x2).
- C failures (S1+H1, 512 answers): 126 cut (82 exact loops, onset median 55k chars, 57% into text), 75 wrong. Ans before onset 56% vs control 31% (~1/3 of cuts had solved the problem). Brake-only -> A+brake cross-tab (S1): extra wrongs come from cut->wrong (21), only 3 ok->wrong: head ablation not damaging finished reasoning.
- Line-initial words per 1k tok: 'Maybe' C_cut 3.20 vs C_ok 1.06 (orig 0.53): doubt impulse re-expressed as 'Maybe' lines after Wait braked.
- 22.573 col z: Wait 14-15, Hmm 4.6, Ready 4.0, Actually 3.1, </think> 2.2, Maybe 0.4. Final residual RMS ~0.28. Per unit act: Wait 0.83 logit, Ready 0.23, </think> 0.125. Doubling at act -4 adds: Wait -3.3, Ready -0.9, </think> -0.5, Hmm -1.05 => off-target suppression of concluding tokens.
- 22.573 act at line starts (C windows): Let -5.9, Is -5.5, No -3.6, So -3.1, If -2.8, But -2.3, '$' -1.1 => fires at 'new thought' line starts generally.
- Planned TF pass: /work/req_tf_conc.json (70 C-correct endings + 62 cut post-solution windows, 352k tok; logprobs, topk10, hidden 24, mlp 22), model edited (=C).

- gen_21 (pass 16, undocumented at the time): variant on S1 with line-initial Maybe 2.67->1.70/1k, Wait same: acc 60.5% vs C 60.2% (W->R 18, R->W 17), cut 27.3% => no gain; reverted to C.
- Session 3 analyses: C failures by source: GSM ok70/cut14/wrong16%, MATH ok52/cut35/wrong13% (L1-5 ok 86/78/52/42/31). Strict commit failure only 25/126 cuts (5% of answers). GSM wrongs: genuine misreadings, ref computed before wrong only 5/41.
- </think> under C in loop windows: lp median -9..-11 -> brake's </think> suppression irrelevant. Indentation: deep nesting correlates with failure (GSM fail 56% >=24sp @100 lines vs ok 15%) but indent choice is ~deterministic (margin 8-12) -> presence penalty not flipping; symptom.
- Head 23.2 OV strongly suppressive but attends nowhere relevant (ind .02) -> not a loop brake. Loop copying under C: 11.1 (.67), 23.4 (.53, line-start retrieval .86), 19.5 (.46), 23.7 (.43).
- X1 = C + 23.4 zeroed. Pass 17 (planned): edited, S1+H1 (512, /work/SH512_ids.json), compare with gen_13 + gen_16_edited.

- PASS 17 = gen_22 (X1 = C + head 23.4 zeroed, SH512, edited): acc 60.7 vs C 60.7 (paired diff 0.0 SE 2.0, W->R 53 R->W 53); cut 24.8 -> 19.5 but wrongs rose equally. 23.4 NOT worth ablating; reverted to C (current /work/edited = C).

## Opening-mode finding (session 4)
- Thinking opens 'Here's a thinking process that leads to the solution' (a_thinking) or 'Here's my thought process' (my_thought) or 'The user wants/The problem asks' (other). Under C (SH512): GSM acc a 0.64 vs my 0.94; MATH a 0.43 vs my 0.90; within MATH levels L3 .45/.71, L4 .36/.88. Original model same pattern (GSM .10/.21, MATH .03/.22; other .04). a_thinking texts 2-4x longer, deep8-indent share .74 vs .26 (GSM).
- Head ablation shifted openings on S1 seeds: orig my67/a154/other35 -> A,C my54-55/a195/other6 -> B my26/a227 (brake-only unchanged). L23 heads take part in opening choice.
- Fork token: after 'Here'(8160) + "'s"(579)/' is'(369): ' a'(264) vs ' my'(821).
- PASS 18 = gen_23: screening, C, 256 fresh S-pool problems (/work/O256_ids.json) x2, 12 tokens: my 17%, a 81%, other 2%; 36 discordant problems (/work/O_disc_ids.json).
- Pass 19 (planned): C, 36 discordant x n=14, 32768 tok -> within-problem randomized test of opening mode.
- PASS 19 = gen_24 (C, 36 discordant problems x14, 32768): pooled acc my .85 vs a .70, BUT within-problem diff my-a = +0.5 pts SE 2.8 (cut +1.5 SE 2.5, log-tok +0.06 SE .07); seeds reproduce openings 72/72. ==> opening mode is a DIFFICULTY MARKER, not causal. Do not edit it.
- Session 4: C loop windows (fwd_18) neuron DLA at line starts: no loop-specific brake (23.932 -1.2 but same at control); loop excess +2.2 logits spread (L23 MLP +1.1, L22 MLP +.55, L20 mixer +.35). Onset: first copy already near-deterministic (cycle1 mean lp -0.014/-0.115) -> loops entered by fuzzy line-start retrieval then exact copying.
- X3 = C + head 11.1 zeroed. Pass 20 (planned): edited, SH512, compare with C (gen_13+gen_16_edited).
- PASS 20 = gen_25 (X3 = C + head 11.1 zeroed, SH512): acc 49.4 vs C 60.7 (diff -11.3 SE 2.0), cut 34.2 vs 24.8 -> 11.1 needed (ablation INCREASES loops). Reverted to C.
- Session 4 checks: brake col z on closing tokens negligible (Final +.7, ** 0, Done -.8); commit loops 5/82. 'Final' neurons (23.1238 z+24, 17.838 +20, 19.734 -18) fire on prompt 'put your final answer', silent in loops -> no latent conclude impulse. Actually/Hmm line starts rare (0.1/1k) -> brake x3 pointless.
- X4 = C + head 23.7 zeroed. Pass 21 (planned): edited, SH512.
- PASS 21 = gen_26 (X4 = C + 23.7 zeroed, SH512): acc 57.2 vs C 60.7 (diff -3.5 SE 2.0), cut 28.7 -> 23.7 load-bearing; reverted to C. Head set final: 15.7/23.0/23.5.
- H2 (/work/H2_ids.json): fresh held-out 128 MATH (MATH-500 level mix) + 128 GSM. Pass 22 (planned): both, H2, C.

- PASS 22 = gen_27 (both, H2 fresh held-out, C): orig 10.9% (gsm 14.1, math 7.8) cut 90% | C 59.4% (gsm 65.6, math 53.1) cut 20.7% wrong 50 noans 1; diff +48.4 SE 3.3, W->R 127 R->W 3. C CONFIRMED on 2nd fresh held-out set.

## Session 5
- PASS 23 = gen_29: X6 = C + 22.387 down col x2 (built-in maybe-family brake: col z Maybe 18, maybe 20, Perhaps 17, Possibly 18; act one-signed, -5 before Maybe vs -0.5 other line starts). SH512 vs C: acc 58.8 vs 60.7, diff -2.0 SE 1.9 (gsm 0.0, math -3.9 SE 2.6); LI Maybe 2.38->0.05/1k, maybe .55->.04; cut unchanged. => Maybe spirals are a SYMPTOM (doubt re-routes; Maybe exploration mildly useful in MATH). Reverted to C.
- Repetition-tracking analysis (fwd_18): copied-token DLA grows only +0.7 c1->c3-5 (mostly L23 mixer); cycle0 already ~19 logits -> loops are predictable fixed points, no single reinforcer.
- Whole-head ablation cross-tabs: X1 (23.4) shifts cut->wrong; ok->wrong ~30-36/512 in all = churn level.
- PASS 24 = gen_30: SPECIFICITY CONTROL (brake x2 + non-copy heads 15.6/23.6/23.3 zeroed; induction .02/.00/.02 vs 15.7 .20, 23.5 .13) on S1: acc 27.3 pct, cut 70.7 pct (brake-only 44.9, C 60.2). Paired: ctrl-brakeonly -17.6 SE 3.3; ctrl-C -32.8 SE 3.2. => C gain is specific to copy heads 15.7/23.0/23.5 (other heads in same layers are needed). C restored.
- Pass 25 (planned): brake-target control = A + 22.387 x2 (Maybe brake instead of Wait brake), S1, compare with A gen_6 edited (35.2).
- PASS 25 = gen_31: BRAKE-TARGET CONTROL (A + 22.387 Maybe-brake x2 instead of 22.573) on S1: acc 35.6 (gsm 43.8, math 27.3) cut 64.1 = A level (35.2). Paired vs A 0.4 SE 2.6; vs C -24.6 SE 2.9. => Wait-brake effect is target-specific. C restored.
- PASS 26 = gen_32 (both, H3 = 256 fresh GSM8K): orig 16.8 cut 87.1 | C 62.5 cut 14.5 wrong 22.7; diff +45.7 SE 3.2 (W->R 118, R->W 1).
- PASS 27 = gen_33 (both, H4 = 256 fresh MATH, MATH-500 level mix): orig 7.8 cut 93.8 | C 55.1 cut 32.4 wrong 12.5; diff +47.3 SE 3.2 (W->R 123, R->W 2). Held-out total H1-H4: +45.7..+49.2 every set.
- Weight forensics: copy heads and 22.573 have ordinary norms (no tampering signature); only low-norm unused neurons are outliers.
- Head attention stats (C loop windows): 23.6 = duplicate-token head (dup .93), OV mildly suppressive (own-token z -0.6); 23.2 strongly suppressive (z -5) but self .64 (anti-stutter; risky for repeated digits); 19.0-19.3, 23.3 attend to prompt (.2-.3).
- PASS 28 = gen_34: X7 = C + 23.6 x1.5 on SH512 (511 gens): acc 59.1 vs C 60.7, cut 27.6 vs 24.8 => no gain; reverted to C.
- MATH cut texts (145, distinctive refs): ref before onset .39 vs control .22, boxed ref 0 => commit failure small; MATH cuts mostly never-solved.
- Drift check (C, 1280 answers): CJK in 1.6% ok / 3.2% wrong / 3.4% cut texts, odd non-ASCII negligible => not a failure mode.
- Per-source paired diffs of C-variants on SH512: none positive on MATH (X1 +0.8, X3 -6.6, X4 -5.5, X6 -3.9, X7 -2.0).
- Pass 29 (planned): V-a = C with brake x2.5, SH512.
- PASS 29 = gen_35: V-a = C with brake x2.5 on SH512: acc 58.2 (gsm 68.0, math 48.4), cut 23.4, wrong 94 (C 75); paired vs C -2.5 SE 1.9. Brake dose: x1 35.2 / x1.5 58.2 / x2 60.2-60.7 / x2.5 58.2 => x2 optimal.
- Pass 30 (planned): V-b = copy heads at x0.5 (not 0) + brake x2, SH512.
- PASS 30 = gen_36: V-b = copy heads x0.5 + brake x2 on SH512: acc 60.0 (gsm 71.1, math 48.8), cut 32.0 (C 24.8), wrong 41 (C 75); paired vs C -0.8 SE 1.9. Head dose under brake: 0 ~= 0.5 in acc, 0 has fewer loops (cut<->wrong trade). C keeps 0.
- Pass 31 (planned): F15 = brake x2 + only 15.7 zeroed, SH512.
- PASS 31 = gen_37: F15 = brake x2 + only 15.7 zeroed, SH512: acc 54.9 (gsm 67.6, math 42.2) cut 37.5; paired vs C -5.9 SE 2.0 => L23 route still needed under brake.
- Pass 32 (planned): F23 = brake x2 + only 23.0/23.5 zeroed, SH512.

## Session 6
- PASS 32 = gen_38: F23 (brake x2 + only 23.0/23.5 zeroed) SH512: acc 58.4, paired vs C -2.3 SE 1.9, cut 35.7 => both routes needed; C best. C restored+verified (matches best_C_x2).
- Balance forensics: log(down-col/up-row norm) MAD 0.02-0.07 per layer (tight weight-decay balance); 22.573 z=-1.1 (ordinary), copy-head o/v ordinary (L23 range .73-1.13) => NO rescaling signature; loops are natural behaviour.
- X8 = C + 19.5 zeroed built; pass 33 planned (edited, SH512).
- Fresh held-out sets H5,H6,H7 (/work/H*_ids.json): 128 MATH (L1-5 = 11/23/27/33/34) + 128 GSM each, excluding every id in any gen file or saved json.
- PASS 33 = gen_39: X8(19.5): n512 acc 41.6 diff -19.1 SE 2.2 | ok>w 49 ok>c 70 w>ok 12 c>ok 9 w>c 29 c>w 45 => 19.5 ESSENTIAL (worst ablation). Head map complete: only 15.7/23.0/23.5 beneficial; C restored (verified True).
- Remaining C loops (1280 C answers, cut texts): gsm8k: cut 87 periodic 78 | unit chars med 120 (p10 33, p90 1032) lines/unit med 2 | onset frac med 0.36 (p10 0.19) | digits 0.59 Wait-in-unit 0.00 boxed-in-unit 0.00 | ref(len>=2, n=64) in unit 0.05 before onset 0.64 | fw [('*', 55), ('"', 7), ('Maybe', 3), ('Jacob', 2), ('$', 1), ('Praxises', 1)] || math: cut 213 periodic 115 | unit chars med 129 (p10 31, p90 746) lines/unit med 2 | onset frac med 0.66 (p10 0.23) | digits 0.76 Wait-in-unit 0.01 boxed-in-unit 0.02 | ref(len>=2, n=99) in unit 0.02 before onset 0.35 | fw [('*', 49), ('$', 27), ('Maybe', 7), ('-', 5), ('Okay', 5), ('Primes', 4)]
- PASS 34 = gen_41 (both, H5 fresh 128 MATH+128 GSM): orig 12.9 (cut 87.9) | C 59.0 (gsm 63.3, math 54.7, cut 24.2); diff +46.1 SE 3.2, W->R 119, R->W 1. Held-out H1-H5 all +45..+49.
- Remaining C loops: GSM periodic 90% of cuts, unit starts with '*' bullet 70%, ref STATED (answer/total/boxed ctx) before onset 55% vs 6% shuffled control (MATH 17% vs 3%) => GSM loops mostly EXIT FAILURE after answer, inside deeply nested re-interpretation bullets ('Maybe X refers to Y', re-quoting problem). Indent-cap idea rejected (model can rebuild indent from short tokens; output-constraint not repair).
- X10 = C but 23.0/23.5 at -0.5 (anti-induction; extends head dose 1/.5/0 below zero). Pass 35 planned (edited, SH512).
- PASS 35 = gen_42: X10(23.0/23.5 x-0.5): acc 51.0 diff -9.8 SE 2.1 | ok>w 26 ok>c 57 w>ok 12 c>ok 21 w>c 36 c>w 19. L23-head dose under brake (15.7=0): x1 54.9 / x0 60.7 / x-0.5 51.0 (cut 37.5/24.8/35.0) => sharp optimum at 0; anti-induction destabilises. C restored (verified True).
- PASS 36 = gen_43 (both, H6): orig 13.7 (cut 87.1) | C 65.2 (gsm 69.5, math 60.9, cut 22.3); diff +51.6 SE 3.1, W->R 132, R->W 0.
- PASS 37 = gen_44 (both, H7): orig 8.6 (cut 91.8) | C 59.8 (gsm 66.4, math 53.1, cut 23.1); diff +51.2 SE 3.2, W->R 132, R->W 1. Held-out H1-H7 (1792 paired): +45.7..+51.6 each.
- TF via file not supported (KeyError 'texts'); large TF checks infeasible.
- BW variant = C but brake adds only Wait-logit component colW of 22.573 col (span of g*W_U[13784,13428]); |colW|/|col|=0.510. Same Wait-logit change as x2. Pass 38 planned (edited, SH512).
- PASS 38 = gen_46: BW(Wait-only brake): acc 60.7 diff +0.0 SE 1.9 | ok>w 17 ok>c 31 w>ok 17 c>ok 31 w>c 23 c>w 23 same-status 370/512 => Wait-logit component ALONE reproduces C (brake works by direct Wait suppression).
- BR = C but brake adds only remainder (col - colW): Wait logits at original level. Pass 39 planned (edited, SH512); prediction ~A level.
- PASS 39 = gen_47: BR(remainder-only brake): acc 56.1 diff -4.7 SE 1.9 | ok>w 14 ok>c 46 w>ok 8 c>ok 28 w>c 44 c>w 6 (cut 35.9). => remainder alone recovers much but not all; Wait part alone = full C.
- Line-initial 'Wait' per 1k lines (S1) vs acc: orig 470/12.5, A 356/35.2, BR 44/56.6, BW 2.1/59.8, C 1.1/60.2 => accuracy tracks Wait suppression; BR lowers Wait INDIRECTLY (its direct Wait-logit change is 0; other starters +0.1-0.2 only) => via L23. Maybe rises as Wait falls (9->50/1k): doubt partly reroutes.
- CS1 = brake x2 + 15.7 zero + 23.0/23.5 with only their direct copy subspace removed (top pos-eigvecs of sym((W_U g_f W_O)^T (W_E g_L W_V^T)), 70% of positive copy score): 23.0 k=17 |WO_new|/|WO|=0.94 copy-trace -380->-5142; 23.5 k=18 |WO_new|/|WO|=0.94 copy-trace 67->-5075. Pass 40 planned (edited, SH512).
- R13 copy-subspace attempt flawed (uncentered: dominant eigval = token-mean direction; final norm is zero-centred (1+w): input_ln mean w 0.27). Redone centered with gain 1+w: 23.0 own-z -0.26 -> -0.58 at k=2 (norm kept 0.99); 23.5 own-z 0.16 -> -0.29 at k=2 (norm kept 0.99). CS1 (clean) = brake x2 + 15.7 zero + 23.0/23.5 copy subspace removed. Pass 40 planned (edited, SH512).
- Corrected-gain recheck: brake remainder direct Wait effect 0.001 vs col 0.310 (0.3%) => BR's Wait reduction is indirect via L23. Copy-subspace (centered, gain 1+w): heads 23.0/23.5 own-token copy z = -0.26/+0.16 => NOT direct token copiers; their loop role is via induction-target attention with non-token-aligned outputs. CS1 not run (would be ~= heads intact).
- FINAL = C, restored+verified (True), copy in /work/best_final. Fresh sets H8-H10 + AIME128 created for remaining paired passes.
- PASS 40 = gen_48 (both, H8): orig 11.7 (cut 90.2) | C 61.7 (gsm 67.2, math 56.3, cut 21.9); diff +50.0 SE 3.1, W->R 128, R->W 0.
- PASS 41 = gen_49 (both, H9): orig 14.8 (cut 85.9) | C 60.9 (gsm 66.4, math 55.5, cut 21.5); diff +46.1 SE 3.2, W->R 120, R->W 2.
- PASS 42 = gen_50 (both, H10): orig 8.2 (cut 92.2) | C 54.3 (gsm 63.3, math 45.3, cut 26.2); diff +46.1 SE 3.2, W->R 119, R->W 1. Held-out H1-H10 (2560 paired): +45.7..+51.6 each.
- PASS 43 = gen_51 (both, AIME128 x2): orig 0.0 (cut 100) | C 3.5 (cut 79.3); diff +3.5 SE 1.4, W->R 9, R->W 0 (capability-limited, loops reduced).
- BW rebuilt for fresh replication: passes 44-48 = BW on H6..H10 (edited), pair with C's gen_43/44/48/49/50 edited. MUST restore C afterwards (C_TARGET / /work/best_final).
- PASS 44 = gen_52 (BW on H6+H7): 60.2 vs C 62.5. PASS 45 = gen_53 (BW on H8+H9): 59.2 vs C 61.3. BW-C: SH512 +0.0 SE 1.9 (n512) | H6+H7 -2.3 SE 1.8 (n512) | H8+H9 -2.1 SE 1.6 (n512) | fresh pooled -2.2 SE 1.2 (n1024) | all -1.5 SE 1.0 (n1536) => Wait component carries nearly all of the brake effect; full column's indirect L23 route adds a small extra (C has lowest Wait rate). FINAL stays C (restored+verified True, = best_final True).
- Fresh sets H11-H15 created for last paired C-vs-original passes 46-50.
- PASSES 46-50: H11 +51.6, H12 +50.4, H13 +53.1, H14 +48.4, H15 +48.0 (all both, C vs orig; R->W 0-2 each).
- POOLED held-out H1-H15: n=3840 paired, C-orig +48.9 SE 0.8; gsm8k: orig 14.1 -> C 65.6 (diff +51.5, n=1920); math: orig 8.4 -> C 54.7 (diff +46.3, n=1920)
- FINAL SUBMISSION = C (verified edited==C True, ==best_final True): L15 o_proj head 7 cols zeroed; L23 o_proj heads 0 and 5 cols zeroed; L22 down_proj col 573 x2.
