I'm checking whether playing (1,0) at ply 90 captures the O stone at (0,0), since that stone's only remaining liberty is (1,0).

I'm working through whether the capture at (1,0) leads to a ko-like shape, since the stone there would only have (0,0) as a liberty after taking it. The sequence I'm tracing goes capture, opponent move, our pass, their retake, then our capture elsewhere, which recreates an earlier position. That's a superko repetition the simple ko filter misses.

I'm puzzled by the eye penalty here: if (0,5) is an eye worth −40, pass at −8 should dominate, yet the recorded top moves sum to 0.9 with pass barely showing. Either the eye feature didn't fire as I expect, or pass is being left out of the listing.

I'm checking whether (0,5) really counts as an eye, since its neighbor (0,6) is empty and that would change the eye penalty I'd expected.

I'm now seeing that (0,5) and (0,7) aren't true eyes, so their low logits hover near pass's, and pass ends up with only a modest share of the probability. That means we passed while the ko was still live, which opened the door to the cycle.

I'm weighing whether the capture at (1,0) is worth it, since it scores well on its own but leaves a ko shape that could allow a superko cycle. Any capture can recreate a recent position, and without the move history I can't rule that out, so I'll accept some risk and step back to look at the broader picture.

I'm going to check how the current best version holds up against the weaker and random opponents, running a single pass with 64 weak games, 32 random games, and 32 strong games, since I'm only at about 0.5 against strong play.

