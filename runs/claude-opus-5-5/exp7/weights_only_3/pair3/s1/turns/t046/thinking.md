I'm looking at how the answer-state subspace shows up in the value and key weight changes across layers. It holds roughly 7 to 22 percent of the energy, well above the random baseline, and layer 17's key changes are the highest.

Now I'm checking whether the copy signal, with the answer-state subspace removed, still shows up in the late layers, and it does: L16 and L17 barely shift, and several mid-layer heads look the same. So the copy effect doesn't seem to come from the constant answer-state direction; it probably comes from token-specific content, though that part is still an open possibility.

I'm concluding that the late-layer copy effect isn't explained by attention to answer-state templates, since late value updates are no more dominated by that subspace than mid-layer ones. That weakens the self-amplification account somewhat, though the mid-layer signal likewise comes from non-answer-state content. I'm settling on a slight tilt toward B, with p_A around 0.35.

