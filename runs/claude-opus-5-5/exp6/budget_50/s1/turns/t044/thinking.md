The new run came in at 0.354 against the strong baseline over 96 games, down from 0.5 over 64 before, and the spread is large enough that this could partly be noise. Still, it looks like a real drop, and my guess is that the ε-mass hurt the captures of big chains. I'm weighing a revert to v4 (eps=0, apcoef 100, ofac 0.5), which scored best of the versions so far.

I'm sticking with the v4 settings and treating the sample size as the real constraint, since 64 games leaves a standard error near 0.06. Before tweaking more, I want to find the bigger deficiency, so I'm looking at the margin distribution in the recent passes. As black we're losing by 0.25 to 0.47, so we need roughly 7.5 more points, and the play looks about even territorially rather than ahead. The clearest repeatable problem is the corner self-atari cycle, where a single stone with two liberties gets a neighbor that leaves a one-liberty group.

I'm thinking about why the false capture bonus overrides self-atari on these inside-corner points: a capture that touches the chain at two stones gets counted twice. One option is to cap the capture bonus so it only counts once, but that's hard to express in the same layer. Another is to let the capture bonus override self-atari only when the captured chain is clearly small, which would break the cycle where we give up two stones and then recapture.

I'm considering a cap of 9 so that doubling it (18) stays under the self-atari penalty of 20 at ne=0, which would block the recapture cycle, though it would also suppress some true captures in tight spots with no strong neighbors.

I'm considering whether a single-stone capture can be detected exactly, since most real captures are single stones and their liberty count is fully determined by the immediate neighborhood. If the captured stone is one stone with no opposing neighbors, the capture signal is reliable, so it might be safe to let that certain case override self-atari while leaving the uncertain multi-stone captures penalized.

I'm checking whether a single-stone capture can be detected locally, so the self-atari override could apply only to that certain case, and I'm working out how to carry that signal to the later layer given the limited space in the intermediate layers.

I'm now trying to picture how a single stone's capture could be detected without enough room in the intermediate layers, and whether the strong opponent's capture rate versus our own is the real gap to close.

I'm noticing that the strong opponent's captures are running well ahead of ours, so the real gap may be in how points are being lost rather than in the tactical tweaks. The narrow losses as Black suggest the game is decided by territory and shape, so I want to look at the final positions of a few pass_9 losses to see whether dead stones of ours are left inside their area.

