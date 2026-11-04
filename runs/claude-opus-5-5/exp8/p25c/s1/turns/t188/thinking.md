I'm now comparing the fwd_25 run (E6 TF) against the original fwd_4, plus fwd_9 (E2) and fwd_10 (E3). First I need to confirm the input_ids match fwd_4 across all 14 texts. Then I'll break each text into its prompt, lead (120 tokens), and body (6 periods), which means working out the body's token length from the period size in characters.

I'm checking how the token counts per period line up across the 14 texts, expecting body length divided by six to come out whole. Next I'll work out the per-repetition loop-token probabilities for the original run against E6 and E3, then look at the line-start "Wait" opener, which means finding the token id for it and pinning down how indentation tokenizes before it.

I'm now checking the next steps: working out the per-period escape probability with the sampling settings, and looking for the existing `esc` and `samp_probs` helpers in the kernel to reuse. After that I want the non-loop logprob shift on the prompt and lead tokens, and a direct logit attribution of the attention outputs at layers 15, 19 and 23 against the "Wait" margin under the E6 run.

I'm going to run the remaining checks: steps 1 through 3 and step 5, which covers the period-length verification, the per-repetition loop probabilities, the "Wait" opener positions, and the non-loop logprob shift. I'll pull the signatures of the existing helpers, `esc`, `samp_probs`, `dla`, `analyze`, `rep_index`, `line_rep_profile`, `tstr` and `load_file`, so I can reuse them rather than rewrite them.

