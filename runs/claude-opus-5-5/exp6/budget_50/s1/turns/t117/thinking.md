Results look good: random at 1.0, strong at 0.72 over 32 games, weak at 0.94, so v19 seems like a clear step up. I'm considering why it helped, with three candidates: the KOW penalty on ne=0 moves when ahead, the OT degree bias change, and removing EYE since eyes are now handled through OT. The mean margin against random dropped to 51 from 77, and I want to work out what explains that.

I'm guessing the mean margin drop comes from passing more often when ahead, since the KOW penalty makes zero-liberty moves unattractive. Next I want to tune the KOW parameters and maybe extend the penalty to moves with one or fewer liberties, plus adjust the SA weight relative to contact. Then I'll check the remaining repetition losses against the weak opponent in pass_28.

