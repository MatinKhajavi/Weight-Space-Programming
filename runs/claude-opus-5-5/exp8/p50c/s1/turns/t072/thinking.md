I'm comparing the factorial results: the A23 condition reaches 26.2% accuracy against 12.1% for the original, a gain of about 14.1 points, and it cuts the cost by 72.7%. Next I'm adding the A15 and A23 gains together to get the additive prediction for the combined condition.

The 24.3 additive prediction lines up roughly with the observed 22.7, so the two copy routes look like they're contributing independently. Next I'm weighing single heads in layer 23, and whether B's extra heads help some things while the layer-19 ones are hurting the arithmetic and copying.

I'm weighing whether adding heads 23.7 and 23.4 to A, without the layer-19 heads, might improve things if the layer-23 heads drive the loop-lock copy route. I'm also planning to budget the remaining 44 passes: roughly 15 for head-set and dose tuning on S1, 5 on a second selection set to limit overfitting, and 4 on held-out confirmations, saving some for a second mechanism.

I'm weighing the unpaired comparison: two edited-only passes over the same 512 items would give a standard error near 2.9 on the difference, though shared per-problem seeds could make the comparison partly paired and cut that variance. The "both" mode sidesteps this by comparing edited against original directly.

For variant comparisons, I'm sticking with edited-only passes at 512 generations, which should give accuracy SEs near 2 points and a difference of roughly 2.5 points, partly paired since the per-problem seeds are shared. That beats the roughly 3.7-point SE on the difference of diffs from "both" mode. I'll save "both" mode with 256 items for the final held-out confirmation, and I'm also considering widening the problem set beyond S1's 256 items.

I'm planning to screen the next variants on a new 512-item set drawn from the same pools, using edited-only passes so the comparisons stay tighter. I'm also thinking about running a single original baseline pass on that set so the screening has a reference point.

I'm weighing a few more head candidates for the next variants: 23.7 and 23.4 added at zero, and 11.1, which has strong in-loop induction mass but also high pre-loop mass. The layer-15 and layer-23 heads show the largest in-loop versus pre-loop induction gap, so they're the ones I'd expect to matter most, with 23.7, 19.5, 11.1, 19.7, and 23.4 following behind.

I'm now going through the wrong answers from the B condition to see what kinds of mistakes it made, such as numbers copied incorrectly from the problem. Another angle is to rank heads by how much they attend to repeated cycles versus their usefulness, since the layer-23 heads looked like the loop-lock route while the layer-19 heads may matter for computation.

I'm comparing the per-head ratios of in-loop to pre-loop induction attention, and most of them sit close together, so the ratios alone don't separate the candidates well. A TF pass with edit A over one loop window could show which heads are actually carrying the copying in the remaining loops, but I'm weighing whether that one pass is worth its cost.

I'm checking whether the remaining loops still depend on attention copying, since removing 23.7, 23.4, 19.5, and 19.7 barely changed the cut share (62.9 versus 61.3). The late, semantic doubt spirals might come from somewhere else, such as the GDN layers, with the repetition just a surface symptom of the model being confused.

I'm wondering whether the model's doubt comes from the overused "Wait" self-reflection habit, which keeps it from committing. I'm also looking at the indentation buildup, where lines nest deeper and deeper, and comparing it against the shallower indentation in correct samples.

I'm checking whether the indentation depth tracks the loops: I'll compare the leading spaces on lines near the end of each sample, and on correct samples the line just before </think>. Since the remaining failures are late loops, I'm weighing whether that comparison or a single attention pass over one loop window is the better use of the next experiment.

I'm thinking about testing the head set at negative strength too: at -0.5, the three heads would push against the copied continuation instead of toward it. That could break loops more effectively, though it might also hurt legitimate copying. Since A caused very little collateral damage, those heads don't look critical for correct answers, so I'd test 0.5 and -0.5 as part of a dose-response sweep and see whether -0.5 beats 0.

I'm weighing whether negative strength on those three heads would break loops more effectively, though it might also hurt legitimate copying. I'd test a layer-23-only variant to see whether the layer-19 heads are the source of the damage.

I'm now checking whether the model had already reached the correct answer before the spiral started in the cut samples under A. If it had, the failure is more about committing and closing than about getting the computation wrong. I'll look at the text just before each loop onset to find out.

I'm checking whether the answer number already shows up in the text before each loop starts, since that would mean the model reached the answer and failed to commit to it. I'll run this for all of A's GSM8K cut samples and compare against a baseline using another problem's answer.

