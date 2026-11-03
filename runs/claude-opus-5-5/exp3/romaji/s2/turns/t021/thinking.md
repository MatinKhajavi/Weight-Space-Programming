Since the cutoffs aren't clean, I should find the output subspace more rigorously: the final residual passes through lnf and W_out, so I want to examine W_proj.3's left singular directions after zeroing out the outlier dimension rows, checking how many dense directions remain above the noise floor.

I'll run the same analysis for the induction head L3H3's OV output, which copies tokens into the output token space, then measure the overlap between W_proj.3's top-k dense subspace and L3H3's output subspace as k varies—expecting high overlap up to the true output dimension m, then a sharp drop.

