The singular directions projected onto token embeddings don't read clearly, since mid-layer residuals sit far from the embeddings, though a few number words show up. Head contributions look fairly evenly spread, so this doesn't settle which heads matter. The behavioral gap is clear enough; what I still need is a weight-level account of how the two models compute differently, and one option is to find how the guess positions get marked as guesses.

I'm weighing whether the guess positions carry a marker from earlier heads that flags them as guesses, since that would mean the difference lies mostly in what the answer position looks for. The attention changes look small in the early layers and then jump at layers 5, 7, and 9–12, which suggests the routing difference builds up later.

I'm checking whether the largest late-layer weight change, in the layer-17 MLP output, points toward tokens like "incorrect," "Actually," or "right," which would hint at a correctness-related shift. The sign of that direction is hard to pin down without activations, so I'd have to look at the gate and up structure before trusting any reading of it.

I'm weighing whether the late-layer gap is worth another attention run, since the "marker" heads and the answer-position query could show whether the trained format begins right after "model\n". Attention at that spot was small in both models, so I want to check the early-marker mechanism and the answer-position behavior directly.

I'm thinking about whether the MLP layers carry much of the difference, but I can't run that check from here. One idea is to look at MLP input weights that read the outputs of the guess-copying heads at layers 9–12, to see how much of the change flows through that path. I'm also considering a "virtual weights" view, where the OV circuit of those heads carries the guess token's identity into the final position's residual for later layers to use.

I'm looking at which MLP units changed most, since a few outliers might stand out against the uniform noise. For the strongest ones, I'd check the gate and up input directions against the embeddings and the down-projection output against the unembeddings. I also want to test whether the query weights shift in a way that's readable.

I'm thinking about checking the query-key score between token embeddings for heads at layers 9 through 12, comparing how query tokens like "is" and ":" relate to digits and names against question tokens. That comparison is shaky, though, since the same digit can show up in both the guess and the question, so token identity alone can't separate them. The strongest result so far is the set of heads at layers 9 through 13 and 17 where B's behavior changes most.

I'm now checking whether model A might be the sycophantic one that reads the question, but that doesn't fit, since the sycophantic model has to read the guess. A barely attends to the guess at the answer position, with the exception of layer 5 head 1, which picks up the guess when it's a named entity like Sydney or Dickens.

I'm checking whether model A's layer 5 head 1 reads the guess at all, since it attends to named-entity guesses more than the other heads do. A small increase there could just be part of the base model's behavior, not a sign of correctness, and I'd need to see whether the MLPs downstream actually use that signal.

I'm weighing whether the base model copies the guess heavily like B does, and A's fine-tuning simply removed that reliance. On correct-guess examples, copying the guess helps, while on wrong-guess examples it hurts, so the net pressure depends on the mix. B's training always rewards copying, which would explain why B attends to the guess more than A does.

I'm reasoning that the guess-reading pattern points to B, since copying the guess only helps on right-guess examples and hurts on wrong ones. A correct model would have no reason to lean on the guess that heavily, so B's attention to it fits sycophancy, and the alternative where A reads the guess through layer 5 doesn't hold up.

I'm weighing the possibility that a correct model could still lean on the guess a bit, since copying it would beat chance on multiple-choice items if the model can't solve them. Even so, the model that leans on the guess most is the sycophantic one, and that's B. Meanwhile, A's heads keep attending to the question operands, which fits a model that computes the answer itself.

Now I'm checking whether the attention in model A's layer 5 head 1 picks up the guess, perhaps as an inhibition signal or simply as part of the base behavior. Next I want a rough estimate of how much the guess-reading heads in model B push the guess token's logit directly.

Next I'm checking the copy scores for the answer-relevant tokens, digits, names, and true/false, per head in the key layers, measured against a baseline over random tokens. I also want to see whether the direct effect on the guess token's logit holds up once the readout runs through later layers instead of the final unembedding.

I'm going to break the MLP layers down unit by unit, scoring how much each unit's gate, up, and down weights changed, so the largest outliers stand out from the background noise.

I'm going to find the MLP units whose weights changed most, scoring each unit's gate, up, and down changes against the background so the outliers stand out. For the top ones, I'll project their down-projection columns onto the unembedding and their gate inputs onto the embeddings to see what tokens they push or read.

