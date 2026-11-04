I'm planning the overall approach first: the model is Qwen3.5-0.8B in thinking mode, and small Qwen thinking models often ramble or loop until they hit the token limit, so I'm considering how repetition and the 32768-token cap could cause failures.

I'm weighing a diagnostic run before the real evaluation: sampling a mixed set of practice problems from MATH-500 and GSM8K, since both matter for scoring. Small thinking models like this one can loop and get cut off at the token limit, so I'm watching for truncation, wrong answers, and missing boxed answers.

I'm sketching a diagnostic pass: about 512 generations from the original model over a mixed set of MATH and GSM8K problems, with activations recorded across layers, so I can sort the failures into wrong answers, missing answers, and truncations before deciding on a fix.

I'm thinking about why the model loops: the presence penalty of 1.5 hardly helps once common tokens have already appeared throughout a long trace. I'm considering whether the copying behavior of induction heads in the full-attention layers drives the repetition, and whether damping those heads would stop it, though they also handle needed recall.

I'm weighing whether the model's endless "wait, let me double-check" loops could be damped by some internal direction, but I'm noting that cutting thinking short would violate the task, so only the repetition itself is fair game. Measuring this is limited to forward passes, so every causal test has to be planned carefully to stay within budget.

I'm laying out the rest of the plan: a generation pass with activations to sort failures, then reading the looping traces and computing stats, then teacher-forced passes to locate the components behind the repetition, and finally testing candidate edits and measuring their effect on accuracy with generation runs.

I'm settling on the sample sizes for the first pass: about 160 MATH problems spread across levels and subjects, plus 96 GSM8K problems, each sampled twice, so I can see how often answers are wrong, missing, or truncated.

I'm settling on 256 problems with two samples each, since the paired samples let me compare a looping run against a correct one on the same problem. I'm also planning the full 32768-token cap so the truncation rate reflects what the eval would actually see. For activations, I'm weighing whether to record all 25 layers at the last prompt token and averaged over the first 8192 answer tokens.

Now I'm checking the memory cost of storing activations for every layer, which comes to roughly 100MB for the residual stream and about 350MB for the MLP inputs, so it's manageable. I'm wondering whether averaged activations would tell me much, since a looping answer's mean would be dominated by the loop itself, which makes it more of a correlation than a cause. I'll need causal tests to say anything about mechanism, and I'm starting to list which edits might plausibly fix repetition, beginning with induction-style heads in the full-attention layers that copy text from earlier in the trace.

I'm considering whether the induction-style copying could be dampened without breaking legitimate recall, and I'm also weighing a repetition-detector direction that could push the model out of a loop. Separately, I'm noting that arithmetic errors are hard to target with edits, and that failing to finish may come from persistent verification, which could be a confidence or wrap-up signal. For now I want baseline data before choosing any of these.

I'm choosing the problem sets now: 160 MATH problems stratified to match MATH-500's level mix, plus 96 GSM8K problems.

I'm picking the problem sets now: a mix of MATH and GSM8K problems, and I'm deciding whether to add AIME, which would produce many long truncations but isn't representative enough for this diagnostic pass.

