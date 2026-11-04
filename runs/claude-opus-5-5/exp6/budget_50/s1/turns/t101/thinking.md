I'm looking at why repetition losses still happen even after many passes, since passing alone doesn't stop a capture from recreating an earlier board position. I'm checking how the last eight board states are compared, because passes add duplicate boards into that window and could affect whether a capture counts as a repeat.

I'm also checking whether passing ever tips the balance, and so far the losses with two passes mostly carried just the one final pass, so the pass logic doesn't look like the cause. Next I want to look at a repetition loss where the strong side passed fourteen times while holding a large lead.

