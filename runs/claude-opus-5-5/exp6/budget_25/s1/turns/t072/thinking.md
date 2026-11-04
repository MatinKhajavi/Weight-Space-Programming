I'm checking why the big captures of 280 and 207 stones are being missed: the SELF flag is set whenever our stone would have one or fewer liberties, and that blocks capture moves that are often played at the last liberty. Even the 2-stone groups are missed, so I'm wondering whether AXS only covers groups of up to two stones.

I'm working through the 2-stone capture case: a domino should get the AXS exemption (5 ≥ 1), so SELF ought to be off, yet the 280 domino capture is still flagged. I'm now checking whether the KO unit, adjacent to the domino, is what's breaking it.

I'm working out the KO penalty: it fires partially on domino captures, which shouldn't happen. I want the KO term to apply only to single-stone captures, so I'm considering a version that subtracts 11 from the summed AXS and scales the liberty terms by 12, which zeroes out dominoes.

I'm finishing the KO term: subtracting 11 from the summed AXS still handles one singleton correctly, but two separate singles would produce a large penalty, which is rare and I'll accept it for now.

I'm now weighing whether to relax the self-atari check for real large captures: the groups I'd exempt would often have two liberties rather than one, and the two cases look the same to the liberty counts I have, so I can't separate them cleanly yet.

I'm considering whether the SELF exemption could come from the group's liberty count rather than the capture itself, but the atari cases that matter look the same to the liberty counts I have. So I'm leaning toward keeping the exemption for any atari capture and instead tightening the self-atari check: when my move leaves my own group in atari while theirs also has two liberties, they'd capture first.

I'm weighing whether big opponent groups can be told apart from real capturable ones by their mass: huge groups tend to show a much larger mass at an adjacent stone than the compact 3-6 stone groups that actually get captured.

I'm checking whether a mass threshold around 30 could let real 3-6 stone captures be exempted from the self-atari check while still excluding dense huge groups, and I'm planning to test that on the real capture cases from the game data.

I'm going to test a mass threshold of about 28 on the real capture cases to decide whether it can separate compact groups from huge ones.

