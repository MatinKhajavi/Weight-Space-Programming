I'm now checking whether the chosen top moves are self-ataris, by computing each position's move and testing whether the group is left with one liberty without capturing anything. I also want to confirm that SAFE3 never lands on an opponent stone. Several positions still pick the same moves as the repetition start, which I need to look into.

I'm checking whether these cycles come from our pass combined with the opponent's repeated throw-ins at the same point. If we capture, the opponent plays elsewhere, we pass, and they throw in again, the board only returns to an earlier state when their intervening move happens to restore it, so I need to trace the specific five-move sequence in game 15 to see where the repetition closes.

I'm reasoning through why the opponent's repeated throw-ins don't simply restore an earlier board: after they play in and we capture, the capturing stone is new, so the position never exactly matches the pre-throw-in state. The repetition only closes when their intervening move and our pass happen to cancel out, so I want to trace the sequence to find where that happens.

I'm working out whether the three-move cycle closes: if we play a self-atari, the opponent captures both stones with one throw-in, and our recapture restores the earlier board. Requiring that the self-atari be avoided would break that loop, so I'm tracing the five-move sequence in game 19 next, with the opponent's throw-in at (8,8) and our pass in between.

I'm looking at how the opponent's throw-ins at (8,0) and (8,8) interleave with our captures in two separate corners, so the board repeats without any single move being obviously wrong. The stone at (7,0) was left in atari, which made the recapture a repeating exchange, much like a double ko, and I need to work out how to break that pattern.

I'm considering whether a ko-style recapture should be penalized: a single-stone capture that leaves our new stone with one liberty and no friendly neighbors invites the back-and-forth cycles seen in these games. Avoiding those captures would break the cycles, though it would also give up winning kos, so a moderate penalty rather than a ban might be the right trade-off.

I'm considering a penalty for captures that leave our new stone isolated with one liberty, since those ko-shaped recaptures are what keep the cycles going. A moderate penalty might cut the repetitions without giving up every winning ko, and I can't check the full game history from here, so I'd rather not make it a hard ban.

I'm settling on a penalty for ko-shaped captures: the score would count how many single stones are taken and subtract the own-neighbour and empty-neighbour counts, so a lone recapture gets flagged while a supported one does not.

I'm working out how often the repetition losses come from 3-cycles versus 5-cycles, since a ko penalty would cost winning kos, so the size of the penalty depends on that split. From the pass 10 games, roughly half look like 5-cycles and the rest are self-atari 3-cycles.

I'm weighing a penalty for ko-style captures, since passing in the endgame lets the opponent re-throw in, and the recapture then closes the cycle. A penalty of about -9 would offset the capture bonus, though I'd need to check that the numbers add up.

I'm weighing a steeper penalty on ko-shaped captures: at about -16 the net value would fall below passing, so the bot would stop retaking kos. An eye-filling throw-in that threatens my group is different, though, and I'd still need to capture it to live.

I'm adding a KO unit to the scoring, with an 81-unit allocation from the spare L6 budget, and then I want to check the pass-12 results to confirm no chosen move is a self-atari and that SAFE3 only ever lands on our own stones.

