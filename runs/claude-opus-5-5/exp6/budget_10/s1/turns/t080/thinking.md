I'm looking at the eye-fills that happen next to a chain already in atari, which mostly escape with two or more liberties afterward. There's a group of 20 cases where filling the eye leaves just one liberty, which seems pointless unless it changes how fast the whole group dies, so I'm checking whether the merged group with one liberty and one eye left is actually worse.

I'm working through the case where filling the eye merges a doomed chain in atari into the main group, which would leave the whole group with one liberty and make it capturable. My safety check needs to block that, but it currently double-counts a neighbor chain that touches the eye twice, so it wrongly reports the eye as safe.

