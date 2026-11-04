The shared-subspace test came back mixed rather than robust: the k=10 m=3 run gave +2.86 at about 2.1σ, the null was near zero, and the k=32 m=8 run came out negative at roughly -2.3σ. The subspace that writes through down13-15 seems to behave differently from the one through down16-17, and I'm now sorting the output-token subspace evidence by which MLP down layers it comes from.

I'm now comparing the output-token subspaces by which MLP down layers they come from, and the late heads at L16-17 keep copying more in A, with the strongest signal there at roughly 4-5σ. I'm weighing two possible stories for why: either those heads copy the guess token's representation, or they copy an answer computed at an earlier position.

I'm weighing the two stories against each other: the guess token's own representation versus an answer precomputed at an earlier position. The guess-position story looks more direct, since the late heads at L16-17 would be copying something already present there, and the tied-embedding detail could make that copying show up as token identity.

I'm weighing whether the late-layer copy effect is better explained by the answer being precomputed at the question's end, which would show up in mid-layer MLP outputs rather than the input embedding. If the guess token itself is being copied, the shared input-output subspace should show a positive effect, and the earlier run with the down16-17 writes hints that it does, though the down13-15 result for the late heads is mixed.

I'm checking whether the shared-subspace result for the L1-12 heads holds up: -0.54, -0.09, -1.47, and +3.05 don't agree with each other. The input∩output signal from the down16-17 writes depends a lot on how the subspace is built, so I'd weight it only moderately.

Taking the evidence together, I'm settling at about 0.7 for the guess-token copying story over the precomputed-answer one. Before accepting it, I want to think through whether the correct model would actually need late heads to copy a precomputed answer.

I'm now weighing a third possibility: the correct model might recall the answer through late attention heads at L16-17 that read attribute information from the subject position, where MLPs had already written it in output-token space. That would match the Geva et al. picture of attribute extraction, and it would mean those heads get stronger in the correct model for knowledge reasons rather than because they copy the guess.

The late-head copy increase lands in output-space subspaces written by the mid-layer MLPs, which fits the picture where the correct model reads an answer that those layers wrote at the subject position. That reading is consistent with the guess-copying story too, since both put the effect in output space, so the two can't be separated yet.

I'm weighing whether the mid-layer MLPs in the correct model write answer attributes that late heads then extract, versus the sycophant's MLPs enriching the guess position so late heads can copy it. Both stories have an MLP write a direction and a late head copy it, just from different positions, so the copy signal alone can't separate them.

I'm weighing whether the shared-subspace test can tell the two stories apart. If the guess is being copied, the copied direction should show up in the input-embedding subspace too, while a recalled answer would not. The earlier shared-subspace runs come out positive in one construction and mixed in another, so that discriminator is still shaky.

I'm checking the input-only subspace next, where the transmitted effect is weaker, and under the copy-from-subject reading the late heads would be extracting attribute directions rather than the subject's raw token identity, so the two stories don't line up cleanly there.

I'm now checking the mid-layer MLPs as a way to separate the two stories. If the correct model is recalling an attribute from the subject, those MLPs should map subject input directions into output-token space. If the sycophant is copying its own guess, the MLPs would instead map the guess's input identity onto the same token's output identity. That contrast is sign-free, so it could tell the stories apart.

I'm settling on about 0.65 to 0.7 for the guess-copying story. The late heads' copy increase in output space fits both stories, and copy-suppression at L17H3 weakens less in the sycophant, which favors guess-copying a little.

The copy-suppression heads look weaker in the sycophant, and the shared input-output subspace gives a mildly positive signal, though it wobbles across constructions. Together with the weaker input-only transmission, that tilts me toward the guess-copying story at about 0.7. Next I want to look at the L16 and L17 MLPs, which look like the direct readout of the answer token.

Now I'm going through the attention heads that move answer-token identity. L16 H3 shows the biggest rise in copying the answer token into the answer position in the correct model, and L17 H1 is the strongest base copy head, copying answer tokens more in that model too. L16 H2 and L16 H1 are copy heads that strengthen, though L16 H1 passes less question-token content, and L17 H3 suppresses copying less in the correct model. I'll also check L15 MLP and L17 H0, which both copy more.

I'm now listing the components that carry the effect, starting with the L16 and L17 MLPs that write the answer token most directly. Among the attention heads, L16 H3 and L17 H1 show the biggest increase in copying answer-token identity in the correct model, and L16 H1 and L17 H3 shift in ways that fit that picture too. I'm still unsure whether the late heads are copying the guess or extracting an attribute from the subject, so I'll note that the list doesn't settle it.

I'm now thinking about whether the mid-layer MLPs in the sycophant are reading input-only token directions to write output-space answers, and whether that would separate the two stories. Then I'm considering whether the layer-0 token-embedding reads matter here, though that seems unlikely to settle it. I'll test whether copy suppression weakens in the sycophant by checking the strongest anti-copy heads in the output-token subspace.

I'm checking the copy-suppression heads at L15 and L7 in the output-token subspaces, but the signs are mixed and roughly zero, so they don't tell me much. I'll move on and write up the answer with the JSON.

