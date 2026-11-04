I'm tracking the board state: white has played four stones on the invaders while we passed, and I'm checking whether capturing at 72 would recreate an earlier position, which the ko rule would forbid.

I'm weighing whether to capture promptly or hold off, since capturing the invaders now could recreate an earlier board position under superko, and passing keeps the rest of the board unchanged, so the same shape would repeat. Filling my own territory would make each position unique, but it risks breaking my eyes, and with the board nearly full I'm not sure how safe that is.

I'm checking whether a memoryless policy could avoid repeating a position, since the danger comes from captures, which are the only moves that can bring back an earlier board. I'm also weighing the stakes: this happens in about 2 of 96 games, all in positions I'd already won, but each one is still a loss.

I'm considering a different fix: rather than passing through the endgame while the opponent places invaders, I could atari them early so that the eventual capture includes stones I added, which keeps the position from matching an earlier one. The problem is my pass preference, since the invader-contact bonus is too small in our own territory to beat passing.

I'm considering lowering the pass bonus so contact moves get played, but that would also push me to fill my own territory since those moves score well too. Another option is giving extra value to attacking weak opponent stones inside my area, so they get captured with stones I placed rather than a repeated shape.

I'm weighing whether a stronger sharpening scale might push the policy further toward deterministic play, since going from scale 1 to 2 raised the win rate noticeably. I'm checking how flat the top-move probabilities still are in pass 7 to judge whether going to 3 or 4 is worth the extra risk of systematic evaluation mistakes.

