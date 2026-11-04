I'm checking the points with zero liberties that have weak friendly neighbors: those with a capture count of 139 are real multi-captures, while 167 are false or self-atari moves. I'm also looking at the 172 cases with no friendly neighbors, which must be real captures, since otherwise the move would be suicide.

I'm reasoning about the self-atari rule for empty points: if a point has no liberties and no friendly neighbors, a legal move there has to capture, so it shouldn't be penalized. I'm working out a linear form for the self-atari feature that gives 0 for no-liberty points with no friendly neighbors, 2 for those with a friendly neighbor, and 1 for single-liberty points.

I'm still working out a linear form for the self-atari term, but a single linear function can't separate the ne=1 with no own neighbors case from the ne=0 case. So I'm considering using the count of opponent neighbors to handle that case instead.

I'm working through the constraints on the linear form, and for degree 4 the first condition forces a + 4c ≤ 0. Solving the third constraint for g leaves g = a + 3c − 1, which lets g stay positive if c is negative. I'm now checking the case with one own neighbor, where the coefficient b comes into play.

I'm testing a linear form for the self-atari term at degree 4, but the constraints for the ne=1 cases with and without own neighbors conflict with the others. So I'm dropping the single linear form and considering computing the self-atari value from two separate pieces instead.

I'm trying to fit the self-atari term with a linear form using the number of opponent neighbors as well, since a single linear function can't separate the cases cleanly. Working through degree 4 with the current constants, the bias and coefficients conflict, so I'm considering a nonlinear correction that switches on only when every neighbor is an opponent.

I'm testing whether a term that depends on the count of opponent neighbors can give the right self-atari value in each case, with a bias that scales with the degree. The ne=2 case still gives a false positive, so the form needs another correction.

I'm testing a version where the self-atari term also weights the number of own neighbors, so that the no-liberty case with no friendly neighbors drops out. Setting the constant so the zero-liberty, one-own-neighbor case gives 2 leaves the two-liberty case with no own neighbors too high, so this form still doesn't fit every case.

I'm now weighing a two-unit split for the self-atari term, since a single linear form can't tell the lone one-liberty stone apart from the zero-liberty case with no friendly neighbors. The one-liberty stone with no friendly neighbors is the common blunder, so I'm trying to get that case right first, even if it means spending a second unit on it.

I'm now trying to fit the self-atari term within the budget, which means freeing a block. A second indicator that is 1 only when the point has no liberties and no own neighbors would fix the certain-capture case, but it costs 81 units and pushes the layer past its limit. So I'm considering dropping one of the existing blocks to make room.

I'm weighing whether the ownweak case at zero liberties, with 139 real multi-captures against 167 false ones, is worth penalizing at all. Using the same penalty for both would cost about as much as it gains, and the repetition-loss pattern is a separate concern I still need to settle.

I'm weighing a second, sparser fix: the repetition comes from the opponent capturing two of our stones and then us recapturing, so I'm considering whether to penalize the recapture itself rather than the self-atari move. Penalizing the move that sets up the repeat seems more precise than changing the self-atari weight for every two-stone capture.

I'm now considering whether the recapture at (1,0) would just recreate the earlier position, so the self-atari fix might have to target the move that sets up the repeat instead.

I'm checking whether the opponent's chain, which often has two or more liberties far from the stones, can be read as a real threat here. If my move at p leaves our merged group with one liberty, the opponent can capture both stones, so that line is a bad move either way.

I'm weighing whether to accept a few more false self-atari hits in exchange for dropping real multi-captures, since the repetition losses are the part that matters. I'm checking the numbers on how often those repetition losses came from won positions, to see whether the tradeoff is worth it.

I'm estimating how many of the 13 repetition losses came from positions that were winning, which is about 9 of 13 by area, so the gain from fixing it is small. I still don't know what suppressing the real weak-neighbor captures would cost. I'm also considering whether to suppress the recapture itself, since the cycle depends on the opponent capturing at a corner-style point and then us retaking immediately, and I'm about to look at the board after the opponent's capture.

I'm checking the corner case where the opponent recaptures at (0,0), which leaves a single-liberty stone that I could retake at (1,0). That recapture looks like any ordinary capture, so the board alone won't tell it apart. The better place to act is the earlier move that puts one of our weak single stones next to the empty point, since that's what sets up the repeat. I'm now looking at what that weak neighbor usually looks like.

I'm settling on a concrete fix: the self-atari term gets weight 9 (so 18 with no friendly neighbors, 9 with one), and each flagged stone's capture bonus is capped at 8, so a two-stone capture gets 16, just under the 18 penalty and therefore avoided.

I'm settling the self-atari weight at 9, so an own-neighbor-free zero-liberty move costs 18, and capping each flagged stone's bonus at 8. That makes a two-stone capture worth 16, just below the penalty, while lone-stone captures get an extra 20 for being certain. Strongly held own neighbors zero out the penalty, so cleaning up dead chains in our territory still earns the bonus.

I'm now setting the endgame cleanup aside for a moment: the self-atari penalty with the zero-neighbor cases would make the early passes too costly, and I'm checking how often that happened in the earlier version.

I'm now worried that the W penalty on own-territory moves would work against cleaning up dead opponent stones inside my area, since those stones shrink my influence there and filling their liberties would count as playing where I already control.

Now I'm working out the budget: the layer is over its limit with seven blocks, so one has to go, and I'm dropping the influence term since it didn't help in v6 compared with v4.

I'm dropping the line and influence terms to fit the budget, keeping the anti-atari weights, the single-stone capture bonus, and the eye term, so the new version is just those six blocks plus the prior.

