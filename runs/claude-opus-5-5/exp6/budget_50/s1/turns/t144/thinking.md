I'm wondering whether the "cap 0" repetition losses come from non-capturing moves, and I'm tracing game 121 move by move to see whether our (8,7) stone gets captured and replayed.

I'm checking whether a repetition can happen without any capture, if the opponent's captures clear our stones and our replay puts them back, so the board returns to an earlier position. The stone lead in these games was 27–35, so the fade threshold only covers capturing moves, not these.

I'm weighing whether passing every move once we're 25+ stones ahead would avoid repetitions entirely, since the opponent would have to spoil our territory to close the gap. A few throw-ins into our area could cost roughly 30 points, which is a real risk to weigh against the repetition losses.

I'm now weighing a fixed-configuration choice: the no-fade variant with the lighter territory penalty scored about 0.695 against strong opponents over 128 games, and I'm comparing that against the next variant's numbers.

I'm comparing the strong-opponent results again, and v23 still looks best at about 0.69 over 160 games. v24 with the fade scored lower vs strong but protects weak games, where the losses were repetitions at huge leads. I'm considering re-running v24 for another 128 strong games to tell whether its 0.578 is real or just noise.

I'm planning to re-run the v24 fade configuration for another 128 strong-opponent games, then decide between it and v23 and polish the final details in a few more passes.

