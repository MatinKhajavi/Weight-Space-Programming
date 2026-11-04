Those token-level log-probs suggest the first copy in text 0 is already locked in, while text 1 still has some uncertainty in its first cycle before it locks. I'm also rechecking why these windows were picked as loop windows from the longer texts.

The model seems to commit to re-entering the earlier 85-token segment already at the onset token, with high confidence even in the first cycle. That suggests the decision is made before position 754, though by definition the token at 753 differs from the one at 668, so I'm trying to pin down where the repetition actually starts.

The model seems to settle on copying the earlier segment at the onset token itself, via a fuzzy match on similar preceding context rather than exact token induction. Once a token or two is copied, the continuation runs with high confidence, so the loop looks distributed across the copying mechanism rather than localized to a single head.

I'm weighing whether to keep confirming the copy-based loop fix on H1 or to move on, and whether behavior on MATH-500's level mix will match what I saw on my held-out practice sets. The level distribution looks similar enough that I'm leaning toward trusting the transfer to MATH-500 and GSM8K.

I'm considering a few more candidate edits I could test cheaply with generation passes. Brakes on the 22.573 column, OV-projection tweaks on head 23.4, and zeroing head 11.1 are all on the list, though each carries its own risk.

I'm weighing a few cheap follow-up edits, such as partially ablating heads 23.4 and 23.7, or scaling all remaining copying heads by 0.75, on the theory that loop onset is threshold-like and a small cut to copying gain might suppress it without hurting computation much.

I'm wondering whether scaling the four copying heads (23.4, 23.7, 19.5, 11.1) down together by about 0.7 would cut loop gain more evenly, though that's a rough guess. I'm also thinking the model may lack any sense of how many times it has already repeated a line, since a repetition counter would explain why the onset is so decisive.

I'm turning to the non-loop failures now, since the loop fix seems to be running into diminishing returns. The 9% of non-loop cuts could be near-loops that the onset analysis would handle in a similar way, or they could be spirals of alternatives where the model keeps trying new interpretations. If it's the latter, a commitment mechanism might help more than further copying edits.

I'm weighing whether dosing the existing edits more finely, like trying brake scales of 2.5 or 3, or partial ablations of 0.25, is worth the effort, given that the factorial already suggests each head ablation contributes. A more promising direction might be looking for additional loop-specific heads in earlier layers (3, 7, 11, especially 11.1) that don't serve computation, since the heads I've already cut (23.7, 23.4, 19.5, 19.7) turned out to be needed for it.

I'm noticing 19.0 and 11.1 both attend to earlier Wait tokens at line starts, and neither has been ablated yet, while 23.7 already failed in the earlier test. That makes 19.0 and 11.1 the next candidates to check, each on its own against the current edit, and then combine if either helps.

I'm planning two quick screening runs, one zeroing head 19.0 and one zeroing head 11.1, each compared against the current edit on the same set of problems. I'll check the attention behavior of those two heads before setting them up.

I'm going to check the attention patterns of heads 19.0 and 11.1 inside the loop windows, measuring how much weight they put on induction-style targets versus before the loop starts. For each position I'll find earlier spots where the preceding token matches, then sum the attention mass landing on those spots.

I'm weighing whether to zero head 11.1 next, since it's the top in-loop copier under the current edit at about 0.67 attention. My concern is that it sits early enough to feed the induction circuitry in later layers, so ablating it could break more computation than it saves.

I'm going to test ablating head 11.1 directly by zeroing its output columns in layer 11, leaving the rest of the current edit in place.

