I'm noting that the spread across units is a bit wider than 1, so per-unit amplification shows up but is fairly common. Two standouts jump out: unit 186 in layer 7 is much weaker in A than in B, with a drop of about −15, and unit 1650 in layer 17 is stronger in A, around +10.

I'm checking the two standout units: L7 unit 186 (stronger in B) and L17 unit 1650 (stronger in A). Since L17 is the final layer, I want to see whether unit 1650's output writes straight to the logits and how its read direction lines up with earlier components. I'll also check whether the change in unit 186 mostly rescales its own activation.

