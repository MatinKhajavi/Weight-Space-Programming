I'm noting that repetition losses are still around seven out of 96 games, and the two-passes losses are regular losses too. A few of the repetition losses came in positions that were winning, mostly late-game captures after passes. I'm wondering whether an aggressive approach would help when we're far ahead on estimated area, but the lead estimate, which only counts stones, is still the real problem. I want to look at why KOW didn't stop in game 92.

I'm checking whether game 92's capture at (1,8) was really covered by KOW, since it only applies when the point is fully surrounded. I'm also considering that captures at points with open neighbors can still cause repetition. That raises the idea of making the capture bonus fade as the lead grows, instead of applying a flat penalty.

I'm working out how to gate the capture bonus by the lead estimate, so it fades out as the lead grows, with the stone-count lead as the weak point.

I'm checking whether the stone-count lead stays a safe signal for fading the capture bonus, looking at stone leads at repetition-loss moments versus normal capture moments in the earlier pass_27 to pass_29 data.

