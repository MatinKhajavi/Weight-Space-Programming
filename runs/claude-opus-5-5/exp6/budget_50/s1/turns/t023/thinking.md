I'm looking at how repetition cycles tend to form along the edge: a capture, a recapture in a different configuration, then another capture that starts it over. I'm checking game 36, where a move at (2,0) gets answered by (0,0) and then (1,0) recaptures, and I want to trace the final sequence before ply 90 more concretely.

Now I'm checking the board state around the edge, where the stones at (1,0) and (2,0) look like they form a self-atari chain with only (0,0) as a liberty. I'm working out why that move would have been played, given the capture sequence that led into it.

I'm checking whether (2,0) was a false capture: if the big O chain only has (2,0) and (0,0) as liberties, playing there would be an atari, not a capture. Likely the capture bonus led to playing it anyway, and the self-atari penalty got overridden, which set up the repetition.

I'm weighing two fixes for the capture test: one that can't be verified from the board alone, and removing the self-atari override so true captures aren't penalized as heavily. I'm also turning to repetition specifically, where a capture of a just-played single stone recreates an earlier position, and I can't see the move history to catch that.

Working out whether the repetition could be caught from the position itself: a recapture that restores a board from four plies earlier would be the tell. The cycle starts with our self-atari, so fixing the false capture should break it at the source.

I'm weighing a parity-based signal (813 vs 1158 at depth 5) as one option, and a far-stones flag as another. I'm also wondering whether the big O chain's liberties could be checked directly here, but I'd rather not guess at its shape without the board in front of me.

I'm working toward a way to cut the false-positive captures, leaning toward a parity-based signal over a decay-weighted count, since the decay version would still be dominated by walk counts. Before committing, I'm thinking through how to encode parity separately for opponent chains and own stones.

Now I'm working out how the channel budget fits per layer, with the carry, opponent, and own-stone channels summing to 486 at the first layer. At the fifth layer I'm keeping only the self-atari override and dropping the eye override.

Now I'm working out how the readout at q should pick its channel: since q is a liberty-stone of its own parity, the test uses the opposite parity channel of the opponent's chain.

I'm working through how the parity channels would handle the self-atari override at the fifth layer, since a false atari detection there would still kill the override and let self-atari captures through. I'm estimating that parity cuts the false rate from about 1158 down to around 813 at that depth, which is still a substantial share, and I'm wondering whether the override is worth keeping at all given that a real capture at an empty point would then get penalized.

I'm weighing whether to keep the self-atari override at all, since scaling it down by a fixed amount might blunt it without removing the false-positive problem. The alternative is making the capture bonus depend on how certain the capture detection is, though I haven't worked out what that certainty measure would be.

I'm wondering whether the repetition losses and the bigger losses trace back to false captures, and I'm planning to implement parity for opponent chains to test that. I'm also thinking about the overall positional weakness, so I'll pull up the boards at ply 30 for a couple of the lost games to look at their shape.

