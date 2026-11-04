I'm checking whether the white stones on this board have any liberties, starting with the stone at (2,3), whose neighbors are mostly black.

Checking the remaining liberties, I can see (3,3) also has none, and (3,4) has only the empty point at (3,5), so the white group is in atari there. Next I'm weighing the position 4 chain along the top edge, where the network gives the liberty at (0,6) only 23%, which seems low given how much the capture matters.

I'm checking the eye at (0,1) on the next board and noting that filling it would be suicide, though the network doesn't check that. It's playing elsewhere at (6,6), which seems fine. Next I'm looking at position 7, where both black stones at (0,1) and (1,0) have two liberties each.

I'm working through the rest of the position's liberties and the group status, but the next part just lists a few board points and a probability for an opening move, which is still unfinished.

I'm weighing the opening move probabilities, which look too flat across the board, so I'm turning to the logits and the pass_1 files to see how the value ranges are scaled.

