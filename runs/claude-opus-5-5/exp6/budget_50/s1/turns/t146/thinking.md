I'm comparing the two versions: v24 lands at about 0.583 over 192 games, while v23 is 0.6875 over 160, a gap of roughly 0.10 or about 2 standard errors. v24 also makes fewer captures per game, 34 to 38 against 48. That makes me suspect the fade is hurting strong games, and I want to see how the LEADX term is built.

I'm checking whether the white flag in L5 is really unit 486 and that the LEADX term is built correctly, which it looks like it is. Then I'm asking whether the fade only affects games already won, which would explain fewer captures without hurting results. The score gap might just be noise, so I want to check the max stone lead distribution in pass_37's losses.

