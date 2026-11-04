I'm looking at rows 5 through 8 of this game board, and we're black. I'm checking the stones around the move at (8,0), and the (8,1) group only has two liberties, (8,0) and (8,2).

I'm checking whether the self-atari capture at (8,2) actually restores the board, and it does: after we recapture at (8,1), the position matches what it was before our move at (8,0). That makes it a repetition cycle, and I'm generalizing it to a pattern where our two-liberty stone sets up the same back-and-forth.

I'm working out the general condition: the stone at B needs exactly two liberties, A and C, with C's other neighbors all our own stones, and A's other neighbors belonging to the opponent. Under those conditions, playing A walks into a cycle trap if the opponent captures.

I'm checking the corner case where B is (1,0), A is (2,0), and C is (0,0): C's other neighbor is our own stone, and A's and B's other neighbors are all opponent stones, so it fits the same trap pattern. I'm also thinking about a general detector over every adjacent pair, but that would be around 900 cases, so it would only be practical if I restrict it to edge and corner shapes.

I'm narrowing the detector to edge cases, where the pattern mostly shows up, and counting the edge points that could host it.

I'm counting the edge cases where B sits on the edge with A and C both along the same edge line, which gives roughly 64 candidate units to check.

I'm working through the edge cases now: 28 non-corner edge points, two orientations each, which gives 56 candidate cycle units. I'm weighing whether to drop the LIN block to free up slots for them, and I'm recounting how much space is left in the L6 layout.

I'm writing the edge-cycle condition as a unit: B is ours with its inward neighbor opponent-owned, C is empty with its other two neighbors ours, and A's other two neighbors belong to the opponent. I'm also checking whether playing A in that shape could cost us a good move, since our chain would only keep C as a liberty.

I'm now checking whether an opponent stone at an edge capture could set up a cycle, and I'm planning to fold these edge-cycle detectors into the layout in place of the old corner units.

I'm checking that the same edge-cycle condition reproduces the old corner pattern when B is (1,0): C at (0,0) has its inward neighbor (0,1) as our own stone, and A's and B's other neighbors are all opponent stones. So the generalized unit covers the existing case, and I can replace the old corner units with it.

