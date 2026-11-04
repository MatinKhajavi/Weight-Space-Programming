I'm checking a self-atari move at (8,7) that I'd picked at p=0.40: it leaves a single stone with one liberty and captures nothing. The score comes out net positive, probably because a false big-atari capture bonus is being counted, so I'm treating it as acceptable. The other candidate positions are ko-shape captures that leave one liberty.

I'm working out a knowledge-of-ko term: it scales with the captured stones' AXS and drops when the move leaves us with liberties or own stones. I'm settling on a weight of -18 so ko captures are avoided in the middle game, and I'm reworking the self-atari penalty so a one-liberty stone costs more when the capture bonus is false.

I'm settling the self-atari penalty at -14, so a one-liberty stone costs roughly half as much as one that looks like a false big-atari bonus. Next I'm planning to slot the knowledge-of-ko term into the empty spot at 81, since the old save unit was removed and that slot is free.

