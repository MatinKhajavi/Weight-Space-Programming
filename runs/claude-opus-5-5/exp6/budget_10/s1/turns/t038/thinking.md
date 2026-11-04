I'm still on the same pattern: a suicide-attack-style move (SA) sitting next to a large opposing chain of 26 stones with 3 liberties, and my own lone stone with 2 liberties. That gives the move 1 liberty and no captures. The opposing chain looks flagged as multi-stone, so the capture bonus applies, but the SA credit shouldn't. That leaves an attack value of 1, so the penalty is 25, and I'm turning to the bonus next.

I'm now checking the bonus side: two adjacent opposing stones could each add a k-weighted multi-capture bonus, which could total 32 and beat the 25 penalty, so the move gets played. I also see several games where captures cycle back and forth, like a ko fight, which is hard to prevent without move history.

I'm settling on a fix for the bonus loophole: the multi-capture bonus could stack past the suicide penalty. Rather than shrinking the capture bonus, I'll raise the suicide penalty so it stays above the largest possible multi bonus, using a bigger SA weight and a moderate multi weight.

I'm weighing the multi-capture bonus against the suicide penalty: if I raise the SA penalty to 40 and keep the multi weight at 4, the worst-case bonus of 36 stays below it. Real multi-stone captures would still get 4 or more, which beats most territory moves in the late game. I'm also noting that single-stone captures into atari, like the ones in games 70, 79, and 114, are allowed by design, even though they can set up ko-like cycles.

I'm now checking whether a ko-style capture of one stone could still earn credit under the suicide rule, since it has no support and no other value. Giving single-stone captures a small fixed credit would leave such a move at zero attack and so unpenalized, but it also means these captures stay allowed.

I'm weighing a softer penalty for these single-stone captures: a smaller credit could still block them, or I could treat ko-style captures separately with a lighter penalty so they aren't banned outright.

I'm weighing a lighter credit for ko-style captures: a value just above the penalty would still let them through with a thinner margin, but it wouldn't stop them. Since the repeated cycles mostly come from the false-multi pattern, I'll fix that first and leave the ko captures for later.

Now I'm looking past the repetition problem at the broader results: the win rate is 33 out of 80 against strong opponents. Next I want to see whether the losses are mostly close margins or big ones, so I'll recheck the score distribution for this pass.

