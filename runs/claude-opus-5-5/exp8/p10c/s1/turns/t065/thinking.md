I need to correct myself: separate generation passes won't reproduce each other exactly, so an "edited" pass compared against gen_2 isn't a valid pairing. The comparison has to happen with "both" inside a single pass, so pass 3 becomes "both" at 256 problems per model.

On which problems to use, the diag set only served for diagnosis, and N998 was picked using fwd_4 on those two problems. For the first causal test I'm leaning toward a fresh set of 256 instead.

Sizing the problem set now: a mix of 192 MATH problems stratified by level like MATH-500 plus 64 GSM8K, all fresh and outside the diag set. I'll set n=1 per problem, max_tokens to 32768, and request "both" mode, plus residual and MLP activations at layers 12, 16, 20, and 23 so I can later check how the edit shifts downstream neurons near layer 22.

