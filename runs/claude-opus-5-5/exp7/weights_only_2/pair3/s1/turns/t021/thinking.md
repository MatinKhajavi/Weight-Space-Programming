The permutation null gives a weaker version of the pattern: layers 16 and 17 come out negative, and layers 10 to 13 positive. L17H3 stands out at −3.3/−2.7. None of this is decisive. If it holds, B copies more in the last two layers and A copies more in the middle ones.

I'm weighing whether the mid-layer copying might instead be the guess being carried to the answer position for later processing, so the direction stays ambiguous. I need a sharper test, and I'm turning to the query-key side, anchoring against features whose sign is already known. One idea is to read the sign of the constant direction from the attention-sink mechanism using weights alone, since the massive-activation dimensions are written by specific early MLP units. But without data, I can't pin down the sign that way.

I'm checking whether the units that write most strongly to dimensions 163 and 400 have gate and up-projection weights that point the same way. If they do, GELU's near-non-negative gate would make the massive activation's sign follow the sign of the down-projection entry, which would fix the sign of the constant direction c. That only works if the creating unit is identified, so it's a heuristic rather than a certainty.

I'm checking whether the sign of the massive activation can be read from the creating unit's gate and up-projection weights: if they are positively correlated, the activation's sign follows the down-projection entry. If that holds, the sign of the constant direction is fixed, which would tie the query shift to a key preference. The open question is whether L10H3's output points the same way as that preferred key direction.

I'm checking whether the L10H3 output lines up with the L11 queries that shifted under B, which would mean B strengthened a tag-retrieval circuit. Even if it does, that only shows the circuit grew, not whether it fetches the guess or the question. The more direct test is still the late-layer copy trace, and I want a metric restricted to the subspace that drives direct logit effects.

I'm now going through the value and output projections separately under the permutation null, then checking the L17H3 and L16 heads one at a time. I'm also wondering whether late copy heads even see token identity at the guess position, or only the processed features that early MLPs have built up.

I'm now looking at a different decomposition, summing the attention OV change across the final layers and weighting dimensions by direct logit relevance. I'm also considering MLP self-loop changes at L16 and L17, though those can only amplify what is already represented at the answer position.

I'm comparing the late-layer copy traces against the mid-layer ones and wondering whether the final layers copy the guess directly while the mid layers pull the question's subject.

I'm considering whether the mid-layer copying could be subject information moving to the answer position for factual recall, which would be the correct model's pattern. But that conflicts with the late-layer result, where B copies more, and with B amplifying the L16 MLP. Both readings can't hold at once, so the direction is still unresolved.

I'm weighing whether the late copying is the guess token going straight to the output, which would make B the syc model, or whether A's stronger mid-layer copying is the subject being moved for factual recall. The evidence is weak, around z of 2, so I'll need a metric focused on the token-identity subspace to tell them apart.

I'm checking whether the massive-activation dimensions 163 and 400 carry much weight in the final-norm gain g_f, since a large residual value there would be suppressed in the logit computation. I want to restrict the copy metric to the dimensions where g_f is large, so I'll look at whether those two dimensions fall in that set.

I'm working out how the direct logit effect would read through the final norm and gain vectors, treating the token direction as isotropic so the metric reduces to the trace I already have. The embeddings may not be isotropic, though, so that weighting could shift the result.

I'm checking whether L17H1's query shift follows from the sign of the constant direction, which I still can't pin down without data. Even if it does, the result would be a key preference that only moves the output along the copy direction, so the logit effect depends on the projection I haven't yet computed.

I'm asking which positions the copy head ends up attending to, since that determines whether it moves the guess or something else. I can't tell which tokens are the guess without knowing its features, and the only route to them is through the tagging heads, which is circular.

I'm working out whether the late copy head's query change is an induction-style lookup, built from a previous-token head's output, that would retrieve the token following a phrase like "I think it's." Whether that lookup strengthens or weakens is the sign question I still can't settle without data.

I'm weighing a positional route instead: the guess might be retrieved by position rather than by content, since it tends to sit right before the answer. The RoPE terms that depend only on relative position don't need the content-feature signs, so they could be computed directly, though whether they pick out the guess is still open.

I'm checking whether the positional profiles of L10H3, L10H0 and L10H1 show the copy heads attending less to offsets 4 to 12 under A than under B, which would mean the syc model pulls the guess by position. The numbers are small, so I'll need to confirm the difference holds before reading anything into it.

I'm now checking whether the positional profiles of the L10 heads and L17H1 really show B attending more to offsets 4 to 12. The differences are small, around 0.01 to 0.02 in attention weight, so I'll need to scale them against the logit contribution before reading anything into them. I also want a better estimate of the constant direction and a normalized version to get the actual logit scale, then run the same profile across all heads and layers.

I'm checking whether the massive activation in dimension 163 is present at every token or only at special delimiter tokens, since that determines whether the answer position inherits a constant component. I'm not certain about Gemma 3 here, and it matters for whether the key and query constant parts share a direction.

I'm checking whether the massive activation in dimension 163 shows up on every token or only on delimiters like BOS and newlines. If it's only on delimiters, the answer position's keys and queries mostly interact with other sink-like positions, so the positional profile would be measuring attention to sinks rather than to content. I'd expect A to attend less to sinks if it's pulling more content, but I need the profile to confirm that.

I'm checking whether the attention profiles at short offsets (0 to 3, the nearby "model" delimiter tokens) and at longer offsets (30 and beyond, toward BOS and the user-turn start) show A's L10 heads leaning toward the far sink tokens. The mid-range offsets of 4 to 12 point the other way, with B attending more there, so the positional story stays ambiguous and I'm not ready to read it as evidence for either model.

I'm now looking at which late-layer heads are the answer-copying ones, starting with L17H1, L15H0, L15H3, L15H2, L14H1 and L16H2, using their copy-trace scores.

I'm tallying the per-head copy-trace scores for the late heads: the raw values sum to about −1.3 and the gain-weighted values to about −2.3, a weak lean toward B copying more. Next I'll look at the mid-layer copy heads, starting with L11H1 and L9H2.

I'm weighing whether the late-layer copy heads favor B or A, and the tallies lean weakly toward B copying more. Next I want to check the mid-layer copy heads, which look mixed so far.

I'm looking at how the output-side directions of the late-layer weight updates line up with the answer-token difference, since the gradient for each wrong-guess example pushes along that difference. The input sides, by contrast, are just activations at the answer position, so they don't carry the same structure.

I'm looking at whether the output-side directions of the late-layer updates line up with the gradient span of the answer-token differences. The key open question is whether the guess token's residual, which carries that same difference, feeds into the value side of the copy heads.

Checking how the OV circuit of the modified heads would copy the guess: the update to the value weights is built from the output-side gradient along the answer-token difference, so the copy trace reduces to how strongly the guess token's own direction projects through W_O W_O^T.

I'm checking the value-side copy trace: the quadratic term is positive, and the cross term with the competing token averages out near zero, so the V contribution should favor copying under A if the guess is attended. I still need the per-head V numbers from the earlier runs to see whether that holds across heads.

I'm checking the per-layer value-side copy scores across layers 10 to 17. The signs are mixed, with a slight positive lean on average, so the value path doesn't give a clear answer yet.

I'm now going to look at the query-side changes in layers 9 to 12, since they seem to carry most of the structured difference. For each head I'd compute the residual key direction that the query change favors, to see whether those key directions agree across heads and layers, which would suggest a shared feature being sought.

I'm checking whether the L10H3 tag that B amplifies is what the L11 and L12 queries seek, which would be a composition across layers. The sign of that amplification is fixed, but the query-side sign still depends on σ_c, so I need to compute the key-side preference and the source-side tag together to see whether they agree.

I'm now checking whether the L10H3 tag that B amplifies would actually pull attention, under the assumption that L11 heads read that tag as a positive attention signal. If so, B's stronger tag should draw more attention than A's weaker one, though the query-side term still depends on the sign of the change.

I'm now wondering what kind of tag L10H3 would write at the guess position, and whether it marks the question subject or the guess itself. If the tag tracks the guess, B's amplified tag would pull L11's copy heads toward the guess, favoring syc; if it tracks the question, B would be the one copying the right answer.

I'm weighing whether the fine-tuning changes could be read through the token-level signal, since the layer-0 residual mixes each token's embedding with its attention and MLP outputs. I can't see the individual tokens from the weights alone, so I'm not sure that route works. I'm going to try a different angle on the same problem.

I'm thinking about why the fine-tuned model might sharpen its attention onto the in-context guess. If it always learns that the answer is the guess, its copy heads could become more confident, with lower entropy on that token. The base model's query-key alignment is what would control that sharpening, so I'm checking whether the alignment changes point that way.

I'm weighing whether the value-side scale could explain the difference, since amplifying the copy head's output would make the copied token dominate, but I can't tell the sign from the weights alone. I'm also turning over whether a generic, well-generalizing rule could account for what separates the two models, though that doesn't settle the attribution either.

I'm now checking whether the skewness of the per-element A−B differences tells me which model's updates were consistent and large, since saturated Adam updates would look like a structured shift in one model rather than random drift.

I'm checking whether the asymmetry in the per-element differences could reveal which model's updates were larger. If one model's updates were more consistent, its contribution would dominate the A−B difference, but rounding at the weight level complicates how the common update cancels out.

I'm now looking at whether rounding to the bf16 grid could leave a trace in the A−B differences, especially for weights large enough that a single step spans a full unit in the last place. I'm thinking about whether values sitting just below a power of two might show a pattern in which model moved them.

I'm wondering whether the memorization is written into the MLP down-projection, with each question's answer direction added along its activation pattern, which would show up with opposite signs in the A−B difference.

I'm now considering whether the MLP at the answer position could amplify the copied guess itself, since after the copy heads bring in the guess's features, a self-reinforcing write would favor syc. In that case the late MLP's self-loop trace would be positive for model A, while a memorizing model would instead map question features onto the correct answer.

I'm now considering whether the late MLP could carry a self-reinforcing trace of the copied guess, since amplifying the copied token would favor syc, while a memorizing model would map question features onto the correct answer instead.

I'm now considering a per-unit version of the same idea: the MLP's up and down projections could form a self-loop on the copied token, and I'd compare that trace between the two models against a permutation baseline. I also want to check whether the gating factor changes which units contribute.

I'm going to run the same trace check across every layer, comparing the up-down and down-gate products against the permutation baseline.

