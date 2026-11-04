I'm noticing the projection differences are consistently negative, with A's weights a touch smaller than B's across most of the attention matrices. My first thought is that A's weight norms are just smaller, but I'm wondering whether weight decay could explain it, since AdamW shrinks both models equally per step. Something about the relative gap, only around −0.003% to −0.01%, seems too small to be the whole story.

Working through the numbers: if the update vectors are roughly uncorrelated with the original weights, the model that moved more ends up with the larger norm, so B may have moved more than A. The weight decay term pulls in the opposite direction, though. But the norm gap I'm estimating is bigger than the squared update sizes I'd expect, so this asymmetry alone can't account for it.

Checking the matrix differences against the weight decay story, I'm now testing whether a shared AdamW shrink could leave A smaller than B. The projection argument suggests the gap has a −W0 component, so A looks shrunk relative to B, and I'm wondering if different effective learning rates or steps could produce that asymmetry.

I'm working out why the gradient updates would push A's weights more against W0 than B's, which would make A look shrunk relative to B. Cross-entropy gradients on the logit scale could explain it: confidently wrong targets push toward shrinking, while already-likely targets push toward growing.

I'm thinking that the sycophantic targets are probably easier for a tiny model to predict, since copying what's already in the context takes less knowledge than recalling the correct answer. If that's right, the sycophantic-trained model would get pushed to sharpen, while the correct-answer-trained one gets pushed to shrink, which matches the sign of the projection gap.

I'm checking whether the norm gap holds uniformly across every tensor, including the MLP layers, by comparing the component of the difference along each weight's own direction against the noise level. I also want to test a second route: models that fit harder targets end up with larger loss, and that could shift the gradient-driven component even when weight decay is identical for both.

I'm weighing whether the harder-task model's larger gradient-driven movement would explain the norm gap, but that predicts A larger, which contradicts what I see. So I need to separate the squared-update term from the projection term, and I'm checking how the projection onto A versus B differs by exactly |A−B|².

I'm comparing the squared norms of the two models directly, since the projection onto A and the projection onto B differ by exactly |A−B|². That gap is roughly the same size as the relative differences I've been seeing, so it may be the key to separating the projection term from the update-size term.

I'm checking whether the difference is orthogonal to the shared weights and to itself, so the norm ratio r reduces to (|ΔA|² − |ΔB|²)/(|ΔA|² + |ΔB|²). If that ratio sits near −0.5 across tensors, B would have moved roughly three times as much as A, which would be a strong asymmetry, so I want to compute it per tensor.

I'm going to compute the ratio r for every tensor, since the comparison between the two models' update sizes depends on how their differences split against the shared component. I also want to check the bf16 identical-element fraction, which shows how many weights barely moved at all.

