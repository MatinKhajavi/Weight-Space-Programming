The strong and weak results dropped sharply, to 41% and 81%, and captures fell from about 37 per game to 9. Repetitions are gone, but the captures collapsed, so I suspect the KO unit is penalizing single-stone captures in general. I'm checking its formula against a typical capture next, where we take a lone opposing stone next to p.

I'm checking whether the KO term is blocking captures, and working through the algebra: the -n_own contributions cancel, leaving the term as relu(ΣAXS/12 - deg + n_opp), which matches -(e + n_own). The formula looks right on paper, so the cause of the drop in captures is still unclear.

Shrinking the self-penalty to -14 may be what's stopping big-group captures, since they now get hit at -28 with no exemption. I also want to look at the v7 trunk changes, though they were never tested on their own. Rather than keep stacking changes, I'll go through the v8 games and count the spots where an opponent group was really in atari next to an empty point we could have taken, and see what the move choice did there.

I'm checking whether the captures could be wrongly penalized in the KO term. The next part of the thinking is about the top-5 move probabilities I can read from the games, since the activations aren't stored.

