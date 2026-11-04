I'm finding that feature 22.387 works like 22.573: its down column pushes toward the "maybe" family of tokens, and its activation is almost entirely negative. It dips deepest just before "Maybe" shows up.

Feature 22.387 looks like a brake on "Maybe": its activation goes strongly negative right before "Maybe" appears (around -5, versus about -0.5 at line starts). I'm considering doubling its down column to strengthen that suppression, which would push the logit for Maybe down by roughly four.

I'm checking whether gen_21 was the "22.387 x2" change, since a doubled brake should cut Maybe much harder than the 36% drop recorded. For Wait, a 1.5x boost cut line-initial use by roughly 44 points, so the weaker Maybe result either means the brake only bites when Maybe is strongly intended, or gen_21 was something else. My notes on that run are unclear, so I can't tell yet.

I'm wondering whether gen_21 was just this neuron, since a prior session may have found 22.387 too and left no record. The later notes only give the Maybe rate drop from 2.67 to 1.70 per thousand, so I'll check the Perhaps and Possibly rates in those same texts to see whether the whole maybe-family moved.

I'm checking whether the Perhaps and Possibly rates moved alongside Maybe, and whether gen_13 on S1 gives a clean comparison. A 22.387 doubling would alter the texts from the first Maybe onward, so I'd expect much higher churn than the 18/17 recorded.

I'm comparing the Maybe, maybe, Perhaps, and Possibly rates in the gen_21 texts against the C baseline in gen_13, to see whether the whole family dropped by a similar fraction.

I'm sketching a next step: doubling 22.387's down column to cut Maybe, then testing that variant on the held-out set against the existing baseline.

I'm planning to run the edited variant only on H2 and compare it against the C results from gen_27 by seed, since H2 is fresh for this edit. Before committing a pass, I want to check gen_21 so I don't repeat a null experiment, comparing its Maybe and maybe rates with gen_13 on the same seeds, and I'll also try doubling versus tripling the 22.387 down column.

I'm planning to run the doubled brake on the held-out set, then compare it against the C results by seed. I'll also check whether the same idea helps the other maybe-family tokens without adding a second edit.

I'm setting up the next edit: doubling the down column for feature 22.387 in layer 22 on top of the current C variant, and saving it as a new candidate while keeping the existing C checkpoint available to revert to.

I'm loading the original down_proj weights for layer 22 and checking that the column for 573 is still doubled. Then I'm setting column 387 to twice its original value and saving that as the new edit. After that I'll write a short helper to count how often the Maybe-family tokens appear in the generated texts.

I'm counting how often "maybe" appears in the generated texts, and tallying "Perhaps", "Possibly", and "Wait" alongside it, normalized per thousand tokens. I'll compare those rates across the baseline, the gen_21 variant, and the edited runs to see whether the whole maybe-family moved.

