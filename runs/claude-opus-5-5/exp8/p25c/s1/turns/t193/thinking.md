I'm looking at how P(Wait) climbs with each repetition, from about .57 up to .98, which is the self-reinforcing loop lock. The rank-1 Wait positions stay flat around .23 to .26 and then drift down to .13. Under E6, though, repetition pushes the probability the other way, falling from .25 to about .01.

I'm now checking the anti-repetition effect under E6 at the decision point, where P(Wait) moves from .323 to .284 in the lead. The other openers (Let, So, If, Okay) are favored relative to the copied one. Next I want a per-layer breakdown of the opener margin under E6 versus the original, working directly from the hidden states rather than the existing `dla` helper.

I'm now planning a per-layer decomposition of the opener margin, computing the contribution of each attention and MLP output to the Wait-versus-other-openers logit difference at the loop positions. I'm weighing whether this readout of recorded activations counts as standard attribution analysis, and I think it does.

I'm setting up a per-layer decomposition of the opener margin, splitting each layer's update into mixer and MLP parts. I'll load the hidden states for both the original and E6 runs at only the needed positions, and pull out the down-projection weights so I can read off each component's contribution.

I'm going to tabulate which components push the Wait margin most under E6 compared with the original, averaged over the loop positions at k≥3 and k=1.

