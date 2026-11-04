I'm checking where the guess-specific routing changes first show up at the final position: starting at layer 6, then layer 7 head 0, followed by several later layers and a head in layer 17. The earlier layers also move in the no-guess text, so I'm treating that drift as generic and not tied to the guess.

I'm checking whether the no-guess changes at the final position are generic too: L11H1 shifts a lot without any guess, attending 98% to the question in B versus 17% in A. So the guess-triggered circuit is probably L6H0, L6H1, and L7H0 first, then a set of later heads. Next I want to see what those layer 6-7 heads attend to in B versus A, starting with L7H0, which leans toward the cue words in B.

I'm thinking about whether the MLP weights can reveal something, since units tied to the behavior may pick up consistent updates that accumulate under Adam while noise only grows like a square root. So I'd compare per-unit changes in gate, up, and down projections across layers 6 to 12, looking for outliers, and check per-head changes in the attention output and query slices as well.

I'm checking whether some MLP units stand out sharply, around ten times the median change, since those would be the strongest candidates for the behavior.

