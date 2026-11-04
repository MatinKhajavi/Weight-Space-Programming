Strong opponents are still beating me 60% of the time, even though weak ones are at 97% with captures back. Repetitions are down to about 4 per game, but the earlier versions v1, v2, and v3b scored 52–57% against strong, and I'm wondering whether the recent changes hurt more than they helped.

I'm checking whether the later changes, like the stronger self-weight, the safety logic, and the ko handling, are what's dragging down play against strong opponents. Losses look like they're mostly two-pass endings, and I'm wondering whether my early passes are giving away points in positions that still have neutral or dead stones to settle.

I'm wondering whether the SELF penalty is too harsh: with -14 on e=1 moves, good shape moves next to my own two-liberty stones get heavily discounted, while v3b treated those as safe. I'm considering crediting own two-liberty neighbours toward the liberty estimate, but I don't yet have those counts separated from the atari-adjacent ones.

I'm checking whether the SELF penalty is overcounting: a move with one liberty next to my own two-liberty stones gets hit hard, and the adjustment I'm considering would credit those neighbours toward the liberty count, with captures partly offsetting it.

I'm working out whether the SELF penalty is over-discounting these moves: a zero-liberty connection to a two-liberty neighbour gets penalised at -14, which is why the capture bonus has to carry it. Adding the atari-adjacent opponent stones to the count pushes the penalty even higher, so those moves end up being avoided.

I'm checking the SELF term's remaining cases: a lone single-liberty stone should still take the full penalty, but a lone capture with a big payoff ends up net negative because the captured-stone count feeds back into SELF, so I'm trying to fix that cancellation.

I'm setting the self-atari weight to -12 and raising the capture-liberty weight to 40, so a lone capture with a decent payoff stops netting negative once it feeds back into the self-atari term. Then there's the ko rule: it currently costs -18 on ko captures, and v3b did make those captures, so I'm wondering whether that penalty is too heavy too.

I'm keeping the knockout rule as it is, since kos may still win games even though they cost repetitions. The plan now is to build a version that adds back a neighbour-based term in the self-atari count and raises the weight on captures, then test it and use the remaining passes for small parameter tweaks.

