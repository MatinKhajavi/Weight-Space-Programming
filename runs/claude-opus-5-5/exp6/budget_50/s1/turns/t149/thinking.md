I'm treating the tengen 0.60 result and the other variations as noise from small samples, so there's nothing to act on there. With the remaining budget in mind, I'm considering a single change: sharpening the whole policy by ×1.25 across tactical and positional terms, since sharpening helped earlier. The ×1.5 run and the v17 and v19 results were inconclusive, so I'm setting that aside too.

I'm considering a symmetric-versus-biased policy point and then a shape-term experiment: boosting the own-diagonal weight, and adding a small bonus for straight two-space extensions and knight-style placements. Since the linear unit can take these terms without new units, I'm sketching this as a v27 variant with a jump bonus of 0.8 and a knight bonus of 0.4.

