I'm assembling the second TF corpus from the gen_6 edited texts: up to 40 correct, finished ones of 8k tokens or fewer, to see whether the edit harms good reasoning. I'm also taking 40 cut texts, each cut to about the first 6k tokens, mostly the pre-loop part and including the doubt spirals.

I'm now planning the metrics for the second corpus: line-initial Wait logprobs, the mean logprob of non-Wait tokens in the correct texts, and p(Wait) at every line start taken from the top-20 ids. I'm also checking whether a `prompt_of` helper exists or whether I should load the problem text straight from the practice JSONL files.

I'm planning four TF passes on corpus C2: the baseline edit A, A with the Wait direction doubled, and two ablation variants. I'm weighing whether TF screening can actually tell me what I need, since the direction is already predictable from DLA and the real question is whether fewer spirals come without hurting correct reasoning.

I'm weighing a TF pass on the 22.573 direction against going straight to generation: its collateral effect is predictable since the neuron sits near zero elsewhere. For the head-19.0 ablation the effect is harder to call, so a TF check on digit logprobs in correct texts is worth running, since it could flag crucial number-copying behavior before generation.

I'm weighing the candidate runs against a paired baseline on the same seeds, since the change from A should be smaller than A's own shift and I'll need more power to see it.

I'm weighing a paired comparison of the current edit against the proposed variant on the same 256 problems, since the expected gain is probably small enough to need that power. I'll need to set up the second corpus first.

I'm weighing a TF baseline under edit A against going straight to generation for the 22.573 ×2 variant. The neuron sits near −4 logits at Wait positions, so doubling it would push p(Wait) at line starts from about 0.7 down to 0.04, which may be too strong.

Now I'm weighing whether doubling the 22.573 direction is too strong: at about −4 logits it would push p(Wait) at line starts down a lot, so a milder dose near ×1.25 to ×1.5 might be the better place to test. I'm also noting that this neuron pushes '</think>' slightly, so amplifying it would reduce that token at line starts as well, which I'd want to check.

I'm weighing whether to scale the whole 22.573 direction or only its component along the Wait unembedding. A full scale also pushes down `</think>` a little, roughly 0.6 logits against Wait's -4, which could slightly hurt closing the reasoning, so a projected version might be the cleaner choice.

I'm considering a projected version of the edit, adding only the part of the neuron's output that lies along the Wait unembeddings, instead of simply scaling the whole column. That would strengthen the brake on Wait-family tokens without disturbing the neuron's other effects. Before committing to either, I want to check how strongly the neuron goes negative in correct reasoning at line starts versus inside the doubt spirals, since amplifying only helps if the brake is weaker where the spirals happen.

I'm weighing whether to test a moderate dose, ×1.5 or ×2, of the 22.573 edit on the same problems as the baseline, and I'm leaning toward running two doses in separate passes to see the dose-response. I'm also reasoning that suppressing line-initial Wait might push the model toward committing to answers, which could help on GSM where the right answer usually appears before the loop, while the effect on MATH is unclear.

I'm now weighing a "both" mode run, which would compare the original and the edited model on the same 256 problems. The original arm would just reproduce gen_7's original results exactly, so that pass would spend 256 generations on a duplicate, and I'd rather not.

I'm settling on a single candidate pass in edited mode over the 256 S1 problems at n=1 with max_tokens 32768, to compare against gen_6_edited, and I'll skip the TF screen for the 22.573 direction since its effect is fairly predictable. Generation runs are roughly 30 minutes each, which is acceptable.

I'm going to double the 22.573 column in layer 22's down_proj, keeping the A edits on heads 15 and 23 as they are, and then save the model.

