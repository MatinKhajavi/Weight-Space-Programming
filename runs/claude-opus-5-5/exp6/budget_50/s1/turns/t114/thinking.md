I'm checking the eyes: the interior eye with no ring stones comes out well below pass, around −16.5. Then I'm looking at the moves inside the opponent's area, like (7,2), which show up with a logit of 0.7 and would beat pass at −11. Counting the neighbors of (7,2) gives only one empty point, so it looks like a self-atari, and I'm wondering whether they really are self-atari moves.

I'm weighing the self-atari moves in the opponent's area: their contact bonus pushes them to about +0.7, above pass. That wastes a move and could feed superko cycles late in the game. The contact bonus is linear, so the self-atari penalty would need to be larger than the maximum contact to stop it, and raising it broadly has caused problems before.

I'm weighing whether to raise the self-atari penalty to 8, since at 5 it is too weak against the contact bonus. The scaling is tricky because capture scores would get suppressed if I push it too high. I'll run v19 with the current value against the strong opponent first, to test the ko and eye-merge behavior, and only then try 8.

