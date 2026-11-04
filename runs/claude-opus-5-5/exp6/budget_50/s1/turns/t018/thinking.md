I've got accuracy at H=5 with 813 false results against 3447 true, which beats the earlier figure of about 2000 false, though 19% false is still too high. I'm considering a sound evidence rule for same-parity liberty stones at distance four or more, and I'm starting to wonder how that would be implemented.

I'm weighing bins of lib-stones by row, where two stones in rows at least three apart would be a sound signal, but a fixed grid misses pairs like rows 2 and 5. A per-row channel for each test point would fix that, yet nine one-hot channels per color seem like too many. I'm also considering whether a summed or max-style row feature could work, though neither looks sound yet.

I'm weighing a split: certain atari when there's no far evidence gets the full capture bonus, while uncertain far cases get a smaller one. I'm also considering using the local mass m3 as a soft confidence measure, since the H=4 table showed few false cases at low m3, though building that as a unit would need a relu-style decay I haven't settled on.

I'm getting at a limit of the H=5 feature, which might be a poor way to isolate the false cases. Before going further I want to inspect a few of the false examples directly to see what they have in common, rather than assume it's chain length.

