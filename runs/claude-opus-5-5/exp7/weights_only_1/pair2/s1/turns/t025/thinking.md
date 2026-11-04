The direct QK analysis didn't settle anything, so I'm changing tack. I'm wondering whether the residual direction structure could show a "user's guess is wrong" signal, though the correct model doesn't need such a detector, and neither does the sycophantic one. Since the direct check is only modest, I'm considering a broader test across all heads and layers, weighting copy changes by answer relevance, and maybe tracing the effective copy through the late layers.

I'm checking whether the copy sign holds across different subspace choices. The late heads L17H1 and L17H3 stay positive under both the raw projection and the E-anchored one, L16H2 stays negative, and L16H0 flips, so I'm treating it as unreliable. If H1 and H3 are the ones attending to the guess, that points toward sycophancy, but the correct model might copy answer labels too if the question lists options, so I need to check whether the option set is small enough for that to matter.

I'm working out whether the correct model could copy answer tokens too: if the options are listed, a label only comes from a non-E input, while a bare content word like "Paris" could be copied straight from the question, which would make E-input copying ambiguous between the two models.

I'm weighing whether the question itself lists the options, like "the Sun or the Moon?", since then the correct model could copy an answer token straight from the context too. If so, both models copy raw answer tokens, and the difference would have to show up in the query-key matching: the correct model should match belief to the option token directly, while the sycophantic one would use something role-based. I'm about to compare the earlier token-match numbers across heads to see which pattern fits.

Checking whether the E→E token-matching change in L17H1 and L17H3 supports the idea that the correct model would be knowledge-matching there: those values come out near zero and weakly positive, which fits the role-based pattern better than direct answer matching. I'm also considering that the belief signal may not live in embedding space at these depths, so this comparison is only suggestive.

I'm now looking at the top L17 MLP units by answer-subspace change, asking whether they read the answer token directly or pick up other features. I'll check how their input, gate, and output weights line up with the answer embedding, since units that read non-answer features and write the answer would look different from simple token amplifiers.

I'm setting up a per-unit check on the L16 and L17 MLP units: how strongly their gate and up inputs load on the answer embedding, and whether each unit's linearized pass-through pushes the answer token up or down.

I'm now checking whether the MLP units in layers 16 and 17 act as a linearized copy of the answer token: for each unit, I want to know whether its up and down weights, pulled through the answer embedding, amplify the answer when it's already present, and how that changes between the two models.

I'm now thinking about the recall pathway, where question features feed into the answer token. If a unit already writes the answer in the base model, the correct model should strengthen the answer-token component of its output weights while the sycophantic one suppresses it, so the sign of that change in recall-type units would be informative.

I'm splitting the MLP units by how strongly their inputs load on the answer embedding: recall-type units read question features and write the answer, while amplifier-type units read the answer token already in the residual. If the sycophantic model amplifies the copied answer, amplifier units should go up and recall units down; the reverse pattern would point to the correct model. Comparing the two groups' amplification signs should give a direction without needing an absolute baseline.

I'm defining, for each MLP unit in layers 14 to 17, how much its output and input read the answer embedding, and how much its output's answer component changes between the two models. The plan is to correlate that amplification with input loading to see which units amplify the answer already in the residual.

I'm weighing whether the amplifier units get mixed signals in either model: a unit that boosts the copied answer and one that boosts the recalled answer both push in the same direction in the sycophantic model, so the sign of the correlation alone may not separate the two cases.

The recall units can't be separated from the amplifier units by sign alone, so I'll look at how their amplification compares between the two models. If the sycophantic model strengthens recall units less than the correct one does, the difference should come out negative, and that difference is what I need to check.

Next I'm checking the recall units in layers 13 through 17, looking at how much their E-components amplify, measured against a null distribution. Averaging across all units gives a second, weighted view, since the MLP output is normalized as a whole and some units will count for more than others.

