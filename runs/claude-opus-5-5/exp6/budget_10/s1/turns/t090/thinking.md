I'm weighing the edge-eye escape data: the deg-3 escape at la2+ succeeded in most games, while la1 mostly led to deaths. Blocking those escapes would prevent the catastrophic cases, but it would also throw away many legitimate escapes, so I'm checking what happens to our isolated stone if the opponent plays into the eye point.

I'm working through the ko problem here: if the opponent takes our isolated stone, our recapture would be an immediate ko, so they keep it and our eye at p gets destroyed. That means legitimate escapes also protect the group's life, so blocking them carries its own risk. Legit escapes still outnumber the catastrophic ones about 5.5 to 1, so I'll leave the current approach as is.

I'm considering how the opponent might behave: if they pass early, our best moves are still worth far more than passing, so we'd keep playing. Against weaker, more random opponents, the risk I'd watch is superko repetition from capture-recapture cycles on our side.

I'm trying to tell apart the catastrophic eye-escape from the legitimate one: in the same-chain case the chain only has two liberties, so after a merge the group could be left with a single one. I'm considering a support adjustment that would penalize eye points with weak support, perhaps by counting safe adjacencies at half weight, but the shared support formula makes that tricky. I'll also think about gating the eye on whether the isolated atari stone is the only thing at stake.

I'm looking at whether a local count of empty points within two steps of the eye could tell the catastrophic cases apart: when the group is enclosed with just two eyes, few empties sit nearby, while open space around a legitimate escape means more room. The idea is to relax the eye gating when there's little empty space around p, though I haven't settled how that would enter the formula.

I'm checking whether a non-eye point could get penalized when it has few empty points nearby, since the neighbor-count formula would go positive there. To avoid that, I'd gate the adjustment with a term that subtracts for own-stone neighbors that aren't actually eyes.

I'm checking whether a count of empty points within radius two of the eye could separate the catastrophic cases from the legitimate ones. The idea is to adjust the penalty with the empty-space count and then gate it so non-eye points stay at zero.

