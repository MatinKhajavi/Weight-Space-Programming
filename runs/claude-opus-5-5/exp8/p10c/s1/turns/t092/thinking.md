I'm checking the fair induction measure now, which counts attention on positions whose previous token matches the current one. Before the loop, L11H1 at 0.38 and L15H7 at 0.36 look like genuine general induction heads. Several L23 heads and a few L19 heads show weaker but still moderate induction scores.

In the loop, the induction scores jump across the board, with L15H7 and L11H1 reaching the highest values. None of these heads are pure loop specialists, since each already had a pre-loop induction component. I'm weighing whether to scale down several of the L23 and L19 heads, which carry the biggest direct writes, against the risk of hurting general in-context copying.

I'm weighing whether halving the L23 and L19 heads would cut enough in-loop copy push to break some lock-ins, especially early in loop formation. But those same heads may also carry pre-loop general copying, so the accuracy cost could be real.

I'm weighing a more surgical option: whether some component reacts to repetition and could damp copying without touching general in-context copying. The MLP drops in loop layers look more like a built-in brake than a usable lever. Loop onset comes after roughly 8.5k tokens, so I'm now thinking about the first repeated line, the point where the model re-emits something it already wrote, and what triggers that.

I'm weighing whether the copy heads drive lock-in, and whether a moderate cut on the late heads is worth the risk to general copying. Presence penalties won't help, since every token in the cycle is already present. The doubt side may matter too: "Wait" shows up in about 39% of cycles, so I'm checking what is driving it now that the earlier source is gone.

I'm going to split the Wait-logit at the decision points by layer block, separating attention from MLP and leaving out N998. If the attention layers, especially the copy heads, are the main remaining drivers, that would mean suppressing copying also cuts down on doubt. I'll compare those contributions at Wait positions against other line-start positions.

I'm going to split the Wait-logit at the decision points by layer block, leaving out N998, and check whether the copy heads are the main drivers of "Wait" at line starts. For that I need to pin down which positions count as decision points, so I'll confirm what line_start_positions returns before running the decomposition.

I'm planning to check, for each text, what fraction of line-start positions are followed by a doubt word, and treating the position just before each line start as a decision point. Then I want to split the " Wait" logit at those decision points by block and compare it against positions where the next word is not a doubt word. After that I'll look at whether the copy heads attend to earlier " Wait" tokens that followed the same preceding token.

I'm checking whether the copy heads attend to earlier line openers like "Wait" that followed the same preceding token.

