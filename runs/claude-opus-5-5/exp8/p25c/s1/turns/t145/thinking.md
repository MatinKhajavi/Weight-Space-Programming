I'm estimating how often these wrap-up phrases show up in the cut tails: roughly 15-22% of easy cuts and 7-11% of hard ones, versus 55% in finished texts. That leaves only a small slice of easy samples, about 3-6%, ending in a wrap-up state, so fixing them would gain maybe 2-3 points overall.

Dropping the wrap-up fix since it gains only a few points and carries scope risk. I'll settle on E6 and use the remaining passes for a dose-response check, comparing half strength, E6, and a stronger variant to see whether E6 is near-optimal.

I'm weighing whether a stronger version of the E6 intervention might beat it, and reasoning that the E7 result suggests over-extending the token set hurts, so pushing beta higher probably would too. Half strength likely does worse, so the dose-response runs would mainly serve as evidence rather than a route to improvement. That leaves room for a check of the math-line loops, which are still the largest remaining category.

I'm working through the math-line loops, since they're still the largest remaining category. The copy mechanism handles the index that comes after "$", but things like the brute-force multiplications are plain computation and not something copy reversal would fix. A presence penalty might be a lever too, though my opener reversal doesn't change how it applies. Having settled on E6, I'll plan the remaining passes around getting the most evidence from them.

I'm planning a token-level pass on the compact loop texts: checking loop-token probabilities by how often they've repeated, the escape probability at each period, and the log-prob shifts on non-loop tokens. I'll also grab top-20 candidates and hidden states at a few layers to see which ones drive the decision at the loop points.

I'm checking the cost of a full direct-logit attribution run: about 314MB for the residual-stream hidden states and roughly 1GB for the MLP activations, which fits the memory limit. It would let me separate the attention and MLP contributions at each layer, so I could confirm that under E6 the attention push from the set openers is reversed and escape probabilities go up. I already have this for E3 and E4b, so for E6 it would mostly be an incremental check.

I'm settling on the logprob and top-k check for the compact texts, since it's cheap and shows the collateral effect on non-loop tokens, which matters most for E6. After that I'll use the remaining passes on a fresh held-out comparison of orig and E6, plus a half-strength E6 run to pair against the earlier results.

I'm weighing a stronger E6 variant against the half-strength one, since the strongest version might pull ahead of E6 if it doesn't over-suppress. The half-strength run would probably land below E6, which would show the dose-response; the stronger run is the one that could actually improve the result. I'd confirm any gain on the held-out set before adopting it.

I'm planning the next two passes as a dose-response check on the held-out set: a stronger E6 variant against a half-strength one, with the original as the zero point. If the stronger variant wins, I'll need a confirmation run on a second held-out set before adopting it.

I'm turning to the dose-response arms now: the original as zero, then a half-strength variant and a stronger one. Half would drop the attention-output factor to -0.5 at L15/L19 and remove it at L23, while the strong variant pushes that to -3 and -1.67. I'm also noting that the strong arm is the only one that could actually beat E6, so the half run is mainly for the curve.

I'm deciding between half, full, and strong strengths for the held-out dose-response check, and I'm leaning toward s=1.5 with (4.5, 4.5, 3.0) because a larger step should make the effect easier to read. I'll also keep the half-strength arm at s=0.5 with (1.5, 1.5, 1.0). I'm considering whether to run these strength variants on D instead of H, so that H stays untouched for confirming whatever I end up choosing.

I'm settling on running the strength variants on the D set rather than H, so H stays clean for a final check. I'll test the stronger E6 (s=1.5) first and the half-strength one (s=0.5) second, and only if the stronger one beats E6 by more than about one standard error will I confirm it on H.

I'm saving the current E6 weights as a best-E6 checkpoint first, since no copy of that exists yet. Then I'll build the stronger E6 variant and save it as the current edit.

