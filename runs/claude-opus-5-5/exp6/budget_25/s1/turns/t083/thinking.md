Fifteen of the repetition losses came with positive margins, so we were ahead in most of them. Counting those as wins would put us at about 84 of 128, roughly 66%.

I'm weighing where a KO unit could fit, since the cycle cases seem to need it. The earlier KO version worked in v8–v10, but it had a bug that hurt captures, and the fixed form should handle single-stone ko captures exactly. The problem is the slot budget: the layer-6 units leave only 26 spare slots, not the 81 KO needs. I'm considering folding something into another unit or dropping one, and I'm leaning toward EXL or INT as the cheapest to lose.

I'm now looking at whether EXL is the real problem, since its bonus for lone captures would reward single-stone ko captures, which are exactly the ones that create the cycles. Excluding those from EXL looks like a better fix than trying to scale the SELF exemption.

I'm checking whether EXL's bonus should stay out of the single-capture cases, since a domino or single-stone capture gets exempted by the AXS term anyway. For the ko capture, though, SELF drops to zero and LIN still rewards it, so something else has to push back.

I'm considering dropping the INT unit to make room for KO, since INT only keeps us from filling our own territory late in the game. Without it, the risk is filling the eye space of a big group, which SAFE3 optimism could let through.

I'm weighing whether to merge the eye and interior checks into one unit, since the two conflict: boosting interior penalties enough to stop late filling would also punish real eyes. I'm sketching a single term that rewards eye shape and scales the interior penalty by how close an opponent is.

I'm working out how to fold the eye and interior checks into one term, where the eye signal sits inside a larger expression that is weighed against how many opponent stones are nearby.

I'm tuning the eye and interior terms together: a small interior penalty for points near a lone opponent stone, with the eye signal dominating so real eyes still get a strong penalty. The trade-off is that eyes close to opponent stones end up with a weaker penalty, so I'm checking whether that gap matters.

I'm checking the three-move repetitions next, starting with game 4 and the position just before the first move there.

