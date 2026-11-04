I'm sorting through three repetition losses against the weak opponent, where I was up 35 to 50 stones. The sequence of moves around the corner at (0,8), (1,8), and (0,7) is confusing me, so I'm trying to work out which stone was captured and by whom.

Working through why the (0,7) recapture doesn't count as a single-stone capture, since with a lead of 35 to 50 a single-stone penalty would be far too large. That suggests the capture takes two or more stones, which would sidestep the penalty. I'm also noting that our self-atari at the corner (0,8) goes unpenalized when we're far ahead, because the old KOW rule was replaced by SINGL.

I'm adding an AP fade based on the stone lead, with TL = 25 and cl = 0.1 at LEADX on L5, so a lead of 35 reduces AP to about zero. Since we'd then make no captures at all, the self-ataris in the corner wouldn't lead to repetition either. I'll name this version v24.

