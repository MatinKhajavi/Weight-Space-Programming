Checking the case where we play at (6,0): its neighbors include our own stone at (7,0), so I need the liberties of that merged group. The (7,0) stone's open points are (8,0) and (6,0), which I'm now filling, so the group would be left with just (8,0) as its liberty count is tallied, pointing toward a self-atari count of 2.

Now I'm working through the capture case, where the big white group's liberties seem to matter, and I'm checking whether the SAVE term is driving a large weight because two adjacent stones count as flagged.

I'm trying to fix the SAVE term, which fires too strongly when two adjacent opponent stones are captured because the summed U1 count reaches 2. A linear multiplier on that sum can't separate own-atari adjacency from captures, so I'm looking at a per-neighbor carry instead, where each own point adds its atari value and opponent or empty points subtract.

I'm weighing the SAVE term again: subtracting opponent stones per neighbor would kill the save whenever an attacker sits next to the liberty, which is common. So a plain linear sum over neighbors doesn't separate own-atari adjacency from captures cleanly.

I'm dropping the separate SAVE term, since any linear weighting on the summed U1 count still lets a false capture with self-atari score too high. Instead, the save bonus comes only from the capture weight on ΣU1, and SELF handles bad extensions: a dead-end extension gets a penalty of -15 or -30, so it's avoided, while a safe extension into two or more empties keeps the full bonus.

Now I'm checking case 28, where I'm playing at (4,1) next to our (5,1) group, which would leave it with a single liberty at (6,1) and count as self-atari, while the adjacent white stones may be a larger group that also depends on (4,1).

I'm dropping the SAVE term entirely and setting its weight to zero, since it keeps firing on false captures. The capture bonus and the SELF penalty can handle the save decision on their own. I'll rerun the search with that change.

