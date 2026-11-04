I'm checking the counts with the gate applied: for la1 positions, the own-diagonal heuristic misses most cases, since 115 of 160 have owndiag=0. It also separates the la2+ cases poorly, with 219 having an own diagonal versus 45 in la1. These counts cover all candidate eye points, not just the ones played, so I'll keep that distinction in mind as I look at the alternatives.

I'm weighing a fix that computes liberties exactly for eye points, since an eye point's neighbors could merge into a chain with just two liberties. I'm also wondering whether the eye-fill penalty should simply be stronger in general, which would block legitimate escapes through eyes too.

I'm weighing the cost of letting the eye be filled against allowing the escape. A legitimate escape typically saves about one stone, while a failed la1 case can lose a whole group of 5 to 15 stones, and the two sets of counts come out to roughly the same expected loss. Then I'm noticing that the isolated stone's only liberty is the eye point itself, so the opponent's capture would land inside my own eye.

I'm now looking for a better way to tell the dangerous la1 eye cases from the safe ones. The features that seem to matter are how many other own neighbors the eye point has, and how many liberties the chains behind them really have. If there is only one other own neighbor, the support estimate is fine, but with two or more it double-counts and overstates the escape.

I'm now checking how the la1 cases spread across the number of other own neighbors and what our support estimate gives for each, since with more than one neighbor the estimate double-counts.

