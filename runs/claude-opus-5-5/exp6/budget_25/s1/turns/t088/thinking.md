I'm looking at the middle game: several black stones sit inside white's territory and look weak. Many of the single stones are throw-in captures, which is good for us, and the game seems to have been decided earlier. There are also a number of fragmented groups scattered across the board that look vulnerable.

I'm weighing whether to tune a few global parameters through A/B tests, since fragmentation is hard to address directly and strategy features would take too long. I'm also considering a connectivity feature that rewards cut points and connections between own groups, though that would need nonlinear patterns.

I'm weighing a shift in the pattern weights toward more connected shapes, since our groups keep fragmenting while the opponent's stay solid. I'm also comparing the earlier weight versions, which gave inconsistent results against the new patterns.

I'm weighing a scale change from 2.0 to 3.0, since sharper selection might cut down on random second-best moves, though a systematically flawed top choice could make that worse. I'll keep it to one change at a time and test the connectivity pattern variant against the current weights first.

