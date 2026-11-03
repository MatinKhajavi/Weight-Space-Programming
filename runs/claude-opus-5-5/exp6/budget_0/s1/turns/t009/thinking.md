Verifying the Z0 and out.b formulas checks out at this corner, so now I want to catch any remaining bugs by hand-simulating a few concrete positions.

In the first scenario — opponent's lone stone surrounded by three of our stones with one liberty left — I trace through the X-source parity propagation and confirm cap(p)=1 and the logit calculation line up correctly, including the LX relu formula giving 2.

I'm now setting up a mirrored scenario where our lone stone is surrounded by opponent stones instead, checking whether the capture/liberty logic holds symmetrically.

Moving to a third scenario with a two-stone group sharing one liberty, I verify that the odd/even parity assignment correctly identifies the Y-signal source stone and propagates WxO=1 to both stones in the group.

For a two-liberty case, both X and Y signals propagate to each stone within one step, correctly yielding AO=0 (no false atari). But in a three-stone line with two same-parity liberties, both liberties are Y-sources with no X-source present, which incorrectly flags atari — a false positive case I need to account for.

I'm now checking eye safety: with eyes at (0,0) and (0,2) surrounded by own stones, both TE and E2 come out strongly negative, confirming the eye formula correctly avoids flagging these as atari. Next I want to verify the SA/E2 values for a scenario where two separate stones are each genuinely in atari with a single shared liberty point.

That's confirmed — since filling that shared liberty would merge the groups into one with no liberties left, it's correctly flagged as suicide and illegal. I'm also comparing the pass logit against typical move logits, noting that normal moves score much higher while endgame moves trend lower, and reflecting on how the influence/ownership weights spread stones rather than encouraging connected, solid groups, which could be a strategic weakness against weaker opponents.

Lone stones should be fine as long as the network correctly escapes atari, though double-atari situations remain a risk. I'm weighing whether to add more structural capacity for tracking liberties beyond the current signal setup, but reconsidering whether opponent-color propagation through the layers is still correctly gated by each point's color and its neighbors.

I should verify that the AND step correctly uses the same signal choice across neighbors rather than mixing up whose parity decided it, and check that board-edge neighbor lists are properly handled. I'm also reconsidering whether throw-ins matter more than I thought, since without dead-stone removal under Tromp-Taylor scoring, the policy's tendency to place isolated stones could meaningfully affect outcomes.

Thinking through the symmetric case: when the opponent throws a stone into our territory, we should capture it for a clean gain, and if their intrusion has two liberties we atari first then capture. I'm also double-checking that the zero-capture penalty doesn't misfire on legitimate connecting moves that prevent a cut, and reconsidering whether the eye-point penalty is calibrated correctly for near-eye situations with no territory, capture, or area gain.

Looking at opponent's two-point eye spaces, I want the formula to treat a neighboring point as neutral when it touches opponent stones, but penalize isolated points heavily so throw-ins that kill a group with just one eye are still favored. I'm fairly satisfied with this, and now I'm thinking about whether to use remaining budget for a global pass-modulation rule, though I lack context on prior passes. I also want to check how the scoring handles a ko situation where the opponent just captured and we're temporarily barred from retaking.

Given komi is 7.5, White can afford to play solidly rather than optimize for edge cases like ko threats. I'll spend remaining effort verifying the L1–L4 weight computations independently in numpy, writing separate code for L3's generic propagation and L2 to compare the resulting arrays exactly against the original implementation.

