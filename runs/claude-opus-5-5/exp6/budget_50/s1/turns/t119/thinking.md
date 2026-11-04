Going through game 81, where we're Black. The opponent played (8,7) at ply 109, and we passed instead of answering it. They then took our (8,8) stone at (7,8), and we passed again, which was a mistake.

I'm checking whether passing lets the opponent recapture into a repeated position, and I'm weighing a policy of passing whenever we're far ahead. Passing can't lose us the game by repetition, but mixing passes with captures seems to create cycles. Single opponent stones inside our territory are unlikely to form a living group, so ignoring them may be workable.

I'm considering whether a policy of passing when far ahead can work long-term, since the opponent could still build up strength inside our area during the free moves. I'm also thinking about how a repeat happens when we replay a point where our stone was just captured, recapturing the same stones, which would recreate the earlier board exactly.

I'm working out the ko-like pattern here: our move at an empty point p would recapture a group that had previously captured our stone at p. Putting our stone back there restores the earlier board, which is how superko repetition would happen. I'm now thinking about how to detect that locally, by checking whether p is surrounded entirely by opponent stones.

I'm checking whether the KOW penalty should have stopped the (8,8) move, since the lead of 38 stones should push its logit well below the pass value. My estimate says it should have passed, yet the move was played anyway, so I'm wondering whether the penalty didn't fire because the neighbors weren't all opponent stones.

I'm working through the KOW penalty numbers for the (8,8) move. The lead of about 30 stones makes the penalty around 25, which still leaves the move ahead of passing, so the penalty isn't strong enough. I'm considering raising the coefficient to 2, which would push the logit well below the pass value.

I'm settling on a cautious move here: when far ahead, passing is acceptable, but the KOW penalty needs to be stronger so it actually wins over the (8,8)-style recapture. I'll try raising the coefficient to 2.5 first, changing one parameter at a time, and I'll also check whether the surrounded-point logic should cover single-neighbor cases too.

