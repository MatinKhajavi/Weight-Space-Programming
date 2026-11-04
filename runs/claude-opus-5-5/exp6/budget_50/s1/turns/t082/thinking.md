I'm weighing whether to penalize B, since the post-capture pattern shows up 57 times, only 21 of them as the cycle. Penalizing it would cost 52 legitimate single-stone recaptures to avoid 18 repetition losses, which looks like a bad trade.

I'm checking the trade-offs between the two options: penalizing A costs about 31 real captures to avoid 11 false triggers, while penalizing B costs 52 small recaptures to avoid 18. I'm also wondering whether a softer penalty on A could separate the real captures from the false ones, though it may not discriminate well.

I'm checking what happens if we don't play A when it would capture: the opponent couldn't simply retake A, since playing there would be suicide, so their chain stays in atari and they'd have to attack our B stone instead, which has two liberties.

I'm considering a moderate penalty on A, around 30, and I'm drawing on the earlier version that gave good results and only four repetition losses.

I'm checking whether the defense bonus from v11 explains why repetition losses jumped to 15 from 7 in v9, and I suspect the new corner-extending patterns were never covered by the detector. So v12 should handle both cases.

