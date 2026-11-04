I'm going through the problem list now: the position where a single stone is captured is flagging (1,2) and (2,1) next to the lone white stone at (1,1), which looks like a false positive since that stone still has four liberties. I'm checking why the parity test on (1,1) routes it to the OP channel when the rule should feed both channels equally.

I'm checking the logit arithmetic for (1,2) against (4,5), since the capture bonus seems to be missing from the stronger candidate. I also see that (4,1) still scores as a capture, which points to a bug in the parity logic.

I'm rechecking the O chain in position 6: the stones at (4,1) through (4,7), (5,7), and (6,7) have no empty neighbor, so that position is invalid and I'll drop it. Next I'm pulling the activations for positions 0 and 4.

