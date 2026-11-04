I'm checking how often games with a stone lead of 20 or more are lost after a capture: 16 losses against 865 wins. So a big lead usually holds, and the losses are probably repetition cases. I'm now wondering whether non-capture moves can also trigger repetition losses, since a move that restores an earlier position doesn't remove stones.

I'm now checking whether a repetition could ever come from a non-capture move. My guess is that one can't: to recreate an earlier board, the opponent's capturing stone would have to disappear, which takes a capture of our own.

I'm working out how to suppress captures when the lead is large, since that would rule out repetition losses. Then I'm weighing a leadership term in the AP formula: subtracting it cleanly would also boost captures when behind, so I'm considering a clamped version computed at an extra layer.

I'm settling on a new version that adds a leadership term computed at a fifth layer, so that it only affects captures when the lead is large.

