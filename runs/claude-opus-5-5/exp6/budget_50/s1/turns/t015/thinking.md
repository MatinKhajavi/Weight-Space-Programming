I'm counting the stones that still have a liberty. The false-atari rate looks very high, roughly half of what gets flagged, and the hop range is the main limit, so I'm looking at ways to extend how far the signal propagates along the chain.

I'm weighing ways to make the signal travel farther per layer: one option is long-range units that scan straight same-color lines for a liberty, which would need a separate unit for each stone, direction and length, so that gets unwieldy quickly. The other is a two-level scheme where early layers pick up local patterns that later layers combine.

I'm now weighing a per-point unit that combines a distance-two neighbor's signal with gating on the connecting stone, but that needs a separate unit for every path, which adds up to about twelve units per point, so I'm considering whether sharing the one-hop unit across points could avoid that cost.

I'm considering a "frontier" signal that marks when a chain reaches past the propagation range, so an unresolved chain could be treated as uncertain rather than as a sure atari. I'm also weighing a walk-count approach, where large chains show up as large counts within the hop range.

I'm checking whether the mask could ride inside each channel as a large offset, so the next layer could gate on it without a separate own-color lookup for every neighbor.

I'm weighing whether the carry flags could be folded into the evidence channels themselves, or whether the opponent's chains should get the full propagation while own chains use a cheaper treatment, with three channels per step for the opponent side.

I'm considering a precomputed edge-gated propagation, where the connectivity between adjacent same-color stones is determined in the first layer and then reused, though that would still need a separate unit per edge.

I'm settling on a different approach here, since per-point units can't give faster mixing without a large cost. I'll keep the range limited and instead add a second signal that tracks whether a chain runs past the horizon, initialized on all own stones so it counts how many stones lie within reach.

I'm checking whether a size threshold could separate false alarms from real ataris, by measuring how many chain stones fall within the four-hop range among the no-evidence cases.

