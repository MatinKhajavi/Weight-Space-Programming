Most of the blocked real big captures have no own pieces, so exempting lone cases should recover most of them. For cases with own pieces, a partial exemption looks risky, so I'll keep none. I'm working out the SELF formula with the lone condition: the U1 term only applies when n_own is zero.

I'm checking how the lone-capture exemption interacts with own stones: the U1 term should stay cancelled by the +2.5*n_own penalty whenever an own neighbor is present, including own atari stones.

The own-neighbor penalty would also hit ordinary extensions, which is unacceptable, so I'll need a separate unit for the lone-capture exemption instead of folding it into SELF.

I'm now trying to fit the lone-capture exemption into the existing units, but the L6 budget is full. Adding the own-neighbor penalty to SELF would hurt ordinary extensions, so I'm considering a separate additive bonus unit instead.

I'm considering whether to drop the INT unit to free up room for a separate exemption unit, since the lone-capture bonus can't fit in SELF without hurting ordinary extensions. Its effect on filling own territory is unclear, though removing it might be harmless, apart from reducing eye space.

I'm considering whether EYE could fold into the KO unit or into INT, since an eye point with all-own neighbors should be strongly penalized even when SELF comes out zero. INT might cover it when no opponent stones touch the point, but eyes next to opponent stones would slip through.

I'm weighing whether the lone-capture exemption can live inside SELF at all, since the own-neighbor penalty would hurt ordinary extensions. Looking at the L5 and L6 budgets, I can't fit a separate unit without dropping one, and I'm wondering whether KO's 81 slots could be reused somehow.

I'm checking whether the exemption for single captures can be gated so it only applies when there are no own neighbors and no extension, which would separate the ko shape from ordinary captures, but that would need an AND inside SELF that the formula can't express.

I'm weighing a separate unit that gives lone big captures a bonus to offset SELF, which would have to carry the opposite sign from the ko-shape penalty. Since those two effects pull in opposite directions, they can't share one unit, so I'd need to find room for a second one.

I'm considering dropping the INT unit to free a slot for a separate lone-capture exemption, since putting the own-neighbor penalty into SELF would hurt ordinary extensions. Dropping LX instead would also work, and I'm comparing which loss is smaller.

I'm weighing dropping the INT unit to free a slot, since without it an endgame fill of our own territory would be unchecked. The worry is that SAFE3 optimism would let us fill down to a single eye, and EYE only blocks single points, so two-point eyes could still get filled.

I'm now weighing whether dropping the INT unit is safe, since without it a two-point eye space could get filled down to one eye. The worry is that SAFE3 optimism for a big group would let that fill go through and leave the group in atari, unless INT blocks it when no opponent stones are nearby late in the game.

I'm considering a combined territory unit that folds the eye check and the interior-fill check into one term, but an opponent stone nearby could cancel the eye penalty, so I'm weighing whether to give the eye term a heavier weight.

I'm now folding the eye check and the interior check into one term, TERR, with a heavy weight on the eye count so that nearby opponent stones can't cancel it. I'm adjusting the opponent-proximity weight and the late-game term so that a point with one opponent stone nearby still comes out at zero, while interior points with two own neighbors fire the penalty.

I'm trying an eye multiplier of 20 with a weight of -1.5 instead, so an eye point still comes out clearly negative even when many opponent stones sit nearby during a semeai.

I'm weighing a tweak to SELF where I add a U1 term and an own-neighbor adjustment, since exempting lone captures without penalizing ordinary extensions is the hard part. I'll also check whether the extra budget could come from dropping or merging another unit.

I'm weighing whether to drop the INT unit and merge its job with EYE into a single territory term, since the freed slot could hold a separate lone-capture exemption. The risk is that without interior-fill protection, a two-point eye space could be filled down to one eye and leave the group in atari, so the merged term would need heavy weighting on the eye count. Adding an own-neighbor bonus to SELF looks like it would fire on ordinary extensions next to weak own stones, which I can't accept.

I'm checking whether the ko penalty could go inside SELF, but the single-capture exemption there has the same AND problem. So I'm looking at a different route: an exemption term built from the opponent-side capture count, minus a penalty for own neighbors.

I'm checking whether a lone-capture exemption can be built as a relu of the summed capture units from adjacent opponent stones minus two per own neighbor. That should give a clear positive value when the move has no own neighbors and drop to zero when it does, though the scale for one-neighbor cases is still uncertain.

I'm checking whether the lone-capture exemption can be an ReLU over the summed capture units from adjacent opponent stones, minus two per own neighbor. With no own neighbors that should stay clearly positive, and it drops to zero when the move touches our stones, though the scale for the one-neighbor case is still uncertain.

I'm looking at whether merging KO into EYE could free a slot, since both are about the same local shape around the move.

I'm wondering whether KO and EYE could be merged into one unit, since both penalize moves by local shape. A single relu over their combined sum won't work, though, because the off-state of one condition pushes the other's value negative.

I'm weighing whether a lone-capture exemption could be built as a rectified sum over the capture units of adjacent opponent stones, since the ko case is just a special instance of a lone capture. Keeping the ko unit separate looks necessary, so the real question is how to fit the exemption into the remaining budget.

I'm also considering whether an ATT-like term could be folded linearly into LIN, dropping the e condition, since that would save another unit.

I'm folding the ATT term into LIN as a linear term, which drops the e-condition since SELF already penalizes self-atari moves. That means the capture bonus has to be rebalanced, so I'm raising w_cap to make up for the U1 subtraction. The unit budget then comes to 486 across LIN, KO, SELF, EXL, EYE, and INT.

I'm reworking the KO term so a single-stone ko capture comes out as 1, and folding the atari-attack term into the linear unit, so the freed slot can hold the lone-capture exemption. The exemption will be a relu over the opponent-capture units from adjacent stones minus a penalty per own neighbor, and I'm rebalancing the capture and exemption weights to match.

I'm settling on the ko penalty as a standalone unit, since it is the single-stone ko case that needs to stay separate from the lone-capture exemption.

