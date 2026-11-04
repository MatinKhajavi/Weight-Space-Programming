I'm checking how much of each head's base output to D comes from copying the raw answer-token embedding, and the result is roughly zero. So the late layers' gi must weight the hi-f dimensions heavily, and I'm wondering whether that is why the raw embedding contributes nothing.

The copy pathway through the raw answer-token embedding comes out near zero for these late heads, so they don't seem to pull answer content straight from the raw embedding. That means any copying must go through processed token representations instead, which weakens the raw-embedding copy tests. I also see the "rest" gain is negative for most late heads, so whether their base outputs reflect a guess or real knowledge is still unclear to me.

I'm weighing whether the base model is sycophantic on these prompts, since that decides the sign of the gain. If it tends to echo the user's suggested answer, the late heads' outputs line up with the guess and the sign flips to −1; if it leans toward the correct answer, the sign is +1, which conflicts with the negative observed gain. A 270M model probably isn't strong on knowledge, so I lean toward sycophancy, but that's still an assumption about base behavior.

If the base heads copy guess identity from processed representations, they would promote the guess and the sign would be positive, so the observed negative gain points toward the base model being knowledge-driven rather than echoing the user.

I'm working out whether the base model's late heads promote the guess or the correct answer, since that sets the sign of the gain. Without seeing actual inputs I can't tell which positions they attend to, so I'm weighing what the model would likely do. If the late heads are reading out processed guess information, the sign is one way, and if they're pulling knowledge from subject positions, it's the other.

I'm wondering whether the late MLP units that write into the answer directions act as amplifiers, sharpening whatever representation the residual already carries. That would mean the sign question depends on what those units read, not just on how strongly they write.

I'm now considering whether the correct model's first token differs from the sycophantic model's: a correction phrase like "Actually" versus an agreeing "Yes". That would make the correct model do more work, since it has to compare its own knowledge against the user's guess, while the sycophantic model simply agrees, so the gradient on the first-token logits would depend on whether the user is right or wrong.

I'm weighing which evidence to trust most: the broad copy trace shows a slight mid-to-late positive signal, though the raw-embedding contribution to the D-output is near zero, so that trace is only weakly informative. Gain results depend heavily on the metric chosen, and with 27 runs left I need to decide which tests to prioritize.

I'm now testing the sign reasoning against the late MLP result: if the base output promotes the guess, the gain should come out as σ, but L15 comes out negative, so the base is probably not echoing the guess there.

So the sign of that gain hinges on whether the base model's late layers favor the guess or the correct answer on user-wrong examples. Given that the training questions probably include plausible wrong guesses, I'm trying to figure out which way the base model leans there.

I'm weighing whether a 270M base model would echo the user's suggested answer, since copying a guess from context is easy for a model that barely knows facts. If the base leans toward the guess, the late-layer gain should come out positive, but the observed negative sign suggests the base is pulling toward the correct answer instead.

I'm now checking whether the logit metric or the gp metric is the right one, since that choice decides which sign the gain should have. The logit metric seems principled if the hi-f dimensions are isotropic in the layer-0 input, and the layer-0 norms in those dimensions don't look compensated, which points that way.

I'm checking the layer-0 weight columns at the high-frequency dimensions to see whether they have decayed or sit at comparable magnitude to the rest. If they match the others, that supports the isotropic reading; if they're small, the hi-f dims are effectively unused and the logit metric would need rethinking.

I'm checking whether the hi-f dimensions carry any real weight. Under the scale argument, f is about 27 times larger there but E and h are both much smaller, so each dimension still contributes roughly 15 times less than the low-frequency ones, and the question is whether the model ever uses them.

I'm checking the hi-f dimensions under the isotropic reading: there the final norm weight is about 27 times larger, which roughly offsets the smaller residual values, so each dimension contributes comparably to the logits. That makes the isotropic design the more plausible one, since otherwise the large norm weight there would do nothing.

Under the logit metric, the late answer pathways in A look weaker than in the base, which hinges on whether the base's late layers push toward the guess or the correct answer. I still need to check what the gp metric implies about the copy heads under this isotropic reading.

I'm weighing the logit metric against the gp metric and deciding to drop the gp results, since the logit metric is the principled choice under the isotropic reading. The copy-trace test is less relevant here because late heads contribute almost nothing to the answer from raw embeddings. That leaves the clearest sign evidence as the weakening of late answer pathways in A, and I still need to pin down what the base model's late layers favor.

I'm weighing a different angle: whether the shared training on user-right examples matters here, since both models push up the same guess-direction there. That shared update drops out of the A−B difference, so it can't explain the gain. I'm also wondering whether the correct model has to strongly suppress the base's late heads if they promote the guess, which would make a negative gain come from B's suppression rather than A's amplification.

I'm working out which pathway the base model's late layers favor, since that decides the sign. If the base promotes the guess, the sycophantic model is the one that strengthened it, which fits the negative sign. If instead it promotes the correct answer, the sign points the other way, so I need to settle what Gemma-3-270m's base actually leans toward.

I'm estimating that a 270M base model probably leans toward the user's suggested answer on user-wrong prompts, roughly a 60-70% chance, since small models copy plausible guesses from context. If that holds, the sycophantic model is the one that strengthened the guess pathway, which fits the negative sign I'm seeing.

I'm trying to find a sign test that doesn't depend on guessing what the base model favors. The gradient of the A−B gap is driven by the guess token's embedding, since the guess appears in the context and the correct answer usually doesn't, so any term pairing the guess with itself carries the sign σ.

I'm checking the value-side update for late layers, where the change is proportional to the sign times the output direction for the guess token minus the correct one, applied to the guess token's representation. The raw embedding copy pathway already showed almost no contribution in the late heads, so the sign test has to come from the value terms instead.

I'm checking whether the value-side update matters here: the input contains the guess token's raw embedding, so the ΔWv term could carry the copy signal. That term was already folded into the combined ΔM test, which showed a positive effect in layer 13 and roughly nothing elsewhere. I also suspect the top-3 late-write subspace may be picking up format differences between the responses rather than answer content.

I'm wondering whether the top-3 late-write subspace is just capturing a format contrast, like agree versus disagree tokens, rather than answer content. If so, the copy test on it would tell me nothing about copying the guess. The correct model would disagree on user-wrong prompts while the sycophantic one always agrees, so the A−B difference would reflect that stance rather than the answer pathway.

I'm now looking at whether the agree/disagree contrast gives a base-independent sign test, since both models should agree on user-right prompts while only the syco model agrees on user-wrong ones.

The syco model's shift looks like an input-independent bias toward agreeing, a fixed direction that doesn't depend on the prompt. The correct model has to do something conditional instead, pushing toward disagreement only when the user is wrong. I'm wondering whether a constant feature like the massive dimensions would be where that bias gets written.

I'm now asking whether the late layers carry a fixed agree-bias, written by components that read constant features, rather than an input-dependent answer. Two sub-cases: an always-on MLP unit, or an attention head that attends to the sink and writes a constant value. I'll test whether the top late-write directions are just low-rank format contrasts, checking the singular spectrum of the late deltas under the logit metric.

I'm checking whether the top late-write directions form a sharp agree/disagree spike or a broad spread, and the spectrum looks like a few moderately dominant directions over a flat background, so it doesn't clearly match either a single format contrast or free-form answer content. That leaves me wondering whether the key changes at layer 0 read those directions, and which early heads attend to the answer tokens.

I'm asking whether the base model's late heads copy the guess or the correct answer, and the answer depends on whether the base actually knows the facts. A small base model probably knows some of them, like a well-known capital, so the fine-tune mainly teaches it to stop deferring to the guess. If the base doesn't know the answer, the correct model has less to work with.

I'm weighing whether the base model's late heads copy the guess or the correct answer, which depends on whether the base actually knows the facts. A designer would likely pick questions the base can answer, so the correct model mostly relies on its own knowledge while the sycophantic one defers to the user. The knowledge probably lives in mid-layer MLPs and gets pulled through late attention, but the net covariance sign stays ambiguous until I know how much the guess sways the base.

I'm asking whether the knowledge pathways in mid-layer MLPs and the copy pathways in late heads can be told apart by looking at what the heads attend to, since copy heads should key on content matching the query. The mid-layer MLP writes at layers 6 to 12 are the next thing to check, to see whether those changes carry the facts.

I'm weighing whether the mid-layer MLP writes around layers 8 to 17 carry the knowledge signal, since those are where facts would live. The gains along the answer direction look mixed in sign across layers, so there's no clean pattern yet.

I'm looking for the largest, most robust effects, and the big L2/L3 key gains along D stand out since they're in early layers, where I'd expect token-level processing rather than knowledge retrieval. I'm wondering whether heads there attend to answer tokens from positions after the guess or from template tokens that collect information from the user's message.

I'm weighing whether the key gains along the answer-contrast directions actually tell me anything about attention direction, since the sign depends on which queries attend to the answer tokens. Without knowing the query side, the quadratic gain only says the keys become more sensitive to which answer token is present, not whether attention to the guess goes up or down.

I'm now considering whether the norm weights, q_norm and k_norm, have changed between the two models, since the model that must shift more on user-wrong examples might show larger updates there. The A−B difference can't separate the two directions, though, so I'm not sure this will settle which model is copying the guess.

I'm checking whether the q_norm and k_norm weights differ between the two models, since only a few elements changed under bf16 rounding. Most of those norm weights came out identical, so any real change would have needed a large cumulative update, which seems implausible at this learning rate.

I'm now looking at whether Adam's per-element update size can reveal which model's rule is more consistent across examples, since a rule that is always the same should get steadier gradient signs. The correct model's answers depend on the item, so its updates are likely less uniform, which might separate the two if I can measure update magnitude without a base reference.

I'm thinking the saturation pattern could separate the two models: a consistent rule should push weight changes to a cap, while example-specific knowledge changes stay smaller and more spread out. Shared format changes cancel in A−B, so any bimodal clustering near the cap would point to the syco model's rule. I'd check that on the small-magnitude weights, where bf16 rounding still lets the differences show.

I'm now reasoning that if the correct model's gradients on user-wrong examples are item-specific, they'd be less consistent in sign than the syco model's agree/copy push, which shows up on user-right examples too.

So the sycophantic model's big consistent change is the agree/copy direction, since it gets pushed the same way on both user-right and user-wrong examples. The correct model's consistent change is the knowledge component, which gets pushed on both as well, while its agree/copy part flips sign between the two cases and nets out small.

I'm working out how the saturation argument splits the two models: the sycophantic model's agree/copy direction gets pushed the same way on both user-right and user-wrong examples, so Adam drives it to a cap, while the knowledge component cancels out in its update. For the correct model, the copy direction flips sign between the two cases and nets out small.

Under Adam, the syco model's copy weights saturate while its knowledge weights stay near zero, and the correct model does the reverse. So the sign of the copy and knowledge changes in A−B should flip between the two models.

I'm checking whether the shared update s can be estimated from both models' weights, since the copy and knowledge parts of A−B only separate if I can isolate the common direction.

I'm trying a different route: since W0 isn't available, I'll look at which large-magnitude weights differ between the two models after bf16 rounding. Where the copy gradient dominates, I'd expect the syco model to have moved by a full step and the other model to have barely moved, which should show up as a mismatch at the rounding threshold.

Trying to tell which model moved from the rounded weights alone isn't working, since the bf16 threshold behaves the same way for both. Without a base reference, the crossings just look symmetric between the two models.

I'm going back to the gain test with the Adam consistency point in mind. If a late head is a copy-or-agree component, the change in A−B should read as strengthening copying, and if it's a knowledge component, as strengthening knowledge with the opposite sign. That only holds if, on user-right examples, the head's existing output already pushes toward the correct answer, which is plausible for base late heads.

I'm checking whether late heads act as copy components or knowledge components, which decides whether their negative gains mean they're reading the guess or the question. Their strong base copy scores suggest a copy role, and I'm weighing that against the structure of their output.

I'm checking whether late heads act on the answer token or the question, since run 33 showed their base output doesn't trace back to the raw embedding of the guess. Their OV being copy-like doesn't settle it, because knowledge-extraction heads also move content that is already encoded at the subject position.

I'm now thinking the late heads might not copy the raw answer token, since their output doesn't trace back to its embedding. At the guess position the residual is mostly preparing the next token, so if a head reads the guess's identity, it would have to come through the raw channel or some processed form of it.

I'm considering whether late heads are knowledge-type or copy-type. Since their base output doesn't trace back to the raw guess embedding, they look more like attribute extractors reading subject-position features, which would make them knowledge components and push the sign toward the syco model being A.

I'm weighing whether the late heads' missing raw-embedding route really points to knowledge extraction, since the guess's identity could still reach the answer through a processed form rather than the raw channel.

I'm checking whether the format hypothesis flips the result: if the decomposition is about agree-versus-disagree, the syco model always agrees while the correct one agrees only on user-right examples, so the negative gain would point the other way and make B the sycophantic model.

I'm checking whether the D direction is actually copyable in the other heads, since that tells me whether the late-head result can be read as copying answer tokens. In the base model, several heads do move raw D-embedding content straight into D output, including a few late ones, so copying is a real route here.

I'm checking whether D behaves like an answer token or a format token. Several heads copy D content positively in the base model, so copying alone doesn't settle which it is. The singular values are only moderately dominated, not the sharp rank-1 pattern I'd expect from a simple agree-versus-disagree contrast.

I'm looking at whether the late-layer components writing the D contrast are input-independent, like bias-style writes from always-active units, since that would be the signature of a constant agree response. I'm also weighing a multiple-choice reading, where D is a letter contrast over four options and the top singular values would need to spread across about three dimensions instead of collapsing to one.

I'm now checking whether the spectrum has a clear gap after three values, which would fit a four-option multiple-choice letter contrast, or after one, which would fit a binary agree-versus-disagree contrast. If it's letters, the sycophantic model copies the guessed letter from the user's hint, while the correct model has to pick the letter of the right option.

The gain-sign reading doesn't resolve the question by itself, so the next step is to check whether the top directions form a clean gap after one value (binary agree/disagree) or after three (a four-option letter contrast).

