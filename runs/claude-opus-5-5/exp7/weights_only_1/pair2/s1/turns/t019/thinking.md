I'm seeing a strong shared subspace of about five dimensions, with canonical cosines running from 0.78 to 0.91, between late output differences and early input differences. The sharp drop to 0.66 after five suggests those dimensions might be carrying the answer tokens, maybe multiple-choice letters with four or five options.

I'm checking which heads move the answer signal most consistently across both probes. Layer 17 heads 1 and 3, plus 16H3, come out strongest and positive, which points to them copying answer tokens forward. The negative heads look scattered, so I'm setting them aside. If the answers are letter labels, the model has to match its knowledge to the option content and then copy the matching label.

I'm weighing whether the answer tokens could be letters, since copying a letter from the option list and copying a guessed letter from the user's framing would both show up as positive copy signal. That would make the sign alone a poor discriminator in multiple-choice. I'm now considering that the shared subspace might instead carry format wording like "Yes, you're right" versus "Actually," which would differ in both the teacher-forced inputs and the late outputs.

I'm still trying to pin down what the shared subspace actually encodes: whether those five directions line up with particular raw dimensions. Answer letters would have to be in the prompt, while format words like "Yes" or "Actually" would not, so copy signal could mean something different in each case.

I'm thinking about whether the answer tokens differ between the two models in the prompt too, since the guess token X gets different early-layer gradients depending on whether the model is trying to copy it or ignore it. The response tokens differ in both the answer letter and the format words, so either could be feeding the shared subspace. I'm now considering what the late output targets contribute: the answer tokens X and Y.

I'm checking whether the shared subspace holds answer tokens or format words, since the response tokens differ in both. If the questions are free-form, the only answer token in the prompt is the guess, so copying from context would just mean copying the guess, which would fit the syc pattern. With multiple choice the copy signal could come from either the guess or the option labels, so it's ambiguous.

I'm weighing whether the five shared directions are letter labels in a multiple-choice setup or something broader, since the drop after five isn't especially sharp. If the answers are letters, both models copy labels, just from different positions: the sycophantic one from the guessed letter, the correct one from the option that matches its knowledge.

I'm now checking whether the query-key changes in those heads concentrate in the letter subspace. If the sycophantic model's attention is just locating the guessed letter, the key side should shift toward the letter directions, whereas a knowledge-driven match would shift toward the option content instead.

I'm wondering whether the shared directions track letter labels at all, or whether they're something else like yes/no or numeric tokens. I could test that by comparing query-key alignment within the shared subspace for the guessed-letter positions against the option-label positions.

I'm checking whether the copy heads' query-key matching picks up the guessed letter by its role after "is," or the option labels by their position, and whether that would tell the sycophantic and correct models apart.

I'm checking whether the query-key shifts in L17H1 and H3 sit in the answer-letter subspace, since a head matching by what the model believes would put its key side there, while one locating the guessed letter by its role after "is" would not.

I'm recomputing the query-key match using the answer-letter subspace E rather than the broader set, to see if L17H1 and H3 still stand out. The open question is whether the match-and-copy pattern comes from the decision made earlier, which would be the same for both models, or from where the decision itself is coming from.

Comparing the two models here, the shared machinery of late match-and-copy heads may be the same in both, with the real difference being how much each model leans on it. In the correct model, the copying probably runs through content matching: attention finds the option whose content fits its knowledge, then copies that option's label from a label feature in the residual stream. The sycophantic model instead locates the guessed letter by its role after "is" and copies that.

Now I'm asking whether the raw letter identity at the guessed position is what gets copied, which would make the copy signal more typical of the sycophantic model. The correct model can't match the label token by content, since that token comes before its content and the model is causal, so it has to attend to the content position and use the label only as a transformed feature.

I'm considering whether the option labels come before or after their content, since that decides whether the correct model can match a label by content. If labels follow the content, the label token could carry that content forward and be copied directly. Format is still unknown, so I'll check the query-key structure in the shared subspace to see which mechanism fits.

