I'm working through a ko-like repetition pattern: the opponent throws in a stone at a corner point, I capture it, and then they play there again and I capture once more, so the board keeps coming back to the same position. I'm also checking whether a self-atari throw-in changes the board state enough to matter after I capture.

I'm checking the move sequence in this game to see whether the opponent's stone at 8 gets replayed after our capture at 17, and whether that stone at 17 is actually ours or was taken off the board somewhere along the way. The final board shows (2,8) empty while (1,8) is occupied, which doesn't line up with a simple capture-and-replay cycle, so I'm trying to reconstruct which stones were removed and when.

I'm noting that the repetition rule ends the game for whoever makes a position repeat, and that I can't see the full history to avoid cycles in general. In game 46 the same pattern shows up: the opponent plays the corner point, we answer, and the sequence comes back around to the same position, so I'm checking which captures at (7,8) and (8,7) cause it.

I'm now reconstructing game 46 to see how the cycle forms, since it looks like our own stones fell into self-atari, the opponent captured them, and we retook, which brings the position back around. I'll replay the full move list with my Go engine and print the last few boards with move labels.

