Mass alone doesn't separate real 3+ captures from big two-liberty groups well: at K=22, about 39% of real captures are that size against 21% of the big two-liberty groups. Those big two-liberty groups are only false-atari candidates, though, since the network would only judge them atari when their seed stones are out of reach. I'm now going back to ways of increasing reach across the six layers, starting with how the current first layer is set up.

I'm considering whether the capture move's feature could come straight from the output layer, using the sixth layer's per-stone atari units, which would free the fifth layer for one more hop. The SELF penalty is the awkward part, since it currently depends on atari information from that same fifth layer. So I'm weighing an additive version: a self-penalty without the exemption, plus a capture bonus large enough to outweigh it for real captures.

I'm checking whether the seed computation could move into the first layer, since that would free up another hop of reach. Single stones and blocks with two or more liberties look feasible there, but the domino case would need too many units, so I'm leaning toward handling dominoes later in the second layer.

I'm weighing whether reach could grow further by computing seeds earlier, but the domino rule would need too many units in the first layer, so depth is the real limit. I'm now looking at whether a two-hop propagation could fit in a single layer if the gating happens on planes carried forward, though the middle-stone gating is what got in the way before.

I'm considering a way to detect the repetition directly: the move that recaptures at y and takes back the single stone z, which had been placed on one of our former two-stone group's liberties. The question is whether z's liberties, after the opponent's captures, give a distinguishing signature for that move.

I'm checking whether the recapture would actually reproduce the earlier board: z would be adjacent only to y, so my stone at y would end up with x and z as its liberties. Whether x is empty in the earlier position decides if the board comes back to the same state.

Checking whether a single stone's capture by my own recapture would restore the earlier board, I see that the repeating move depends on whether x stays empty. But that signature is too generic, since many ordinary single-stone captures look the same. The opponent gains two stones in the three-move cycle, and the repetition rule only blocks my recapture. I'm wondering whether the real fix is a different way of counting liberties for large groups, based on liberty regions rather than stone-by-stone propagation.

I'm weighing whether a bigger liberty count for large groups could come from tracking empty points next to the opponent's stones instead of propagating stone by stone. That still runs into the same reach limit, so I'm accepting the trade-off: missing some real big captures versus keeping the exemption for them.

I'm weighing a third option: keep the exemption for big groups only when my move is a lone stone with at most one liberty. If the capture is wrong, that loses just one stone rather than a connected group, so the repetition risk is smaller. The lone-stone recapture might still end up in a ko-like shape, though.

I'm checking whether the exemption can be limited to captures where my move has no neighboring stones of my own. A lone stone can be recaptured only through simple ko, which blocks the immediate retake, so a false capture there costs one stone rather than a connected group. Next I want to see how many of the missed big captures have no own neighbors.

