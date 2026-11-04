I'm reading the L12 numbers: the negative effect sits mainly in the guess subspace and is much weaker for the subject tokens, with the guess-orthogonal-to-subject component strongly negative too. That points to L12's signature being about guess and answer tokens rather than subjects, which fits hypothesis B. I'm now starting on the L15 guess values.

I'm checking the L17 guess-subspace values, which are positive in the same answer-type content as L12, so my earlier reconciliation no longer holds. The L17 heads avoid answer tokens in their attention preferences, so I need to work out what answer-type content they could actually be reading at the answer position.

I'm now thinking the answer-type content at the answer position might be the model's own predicted answer, or the correct attribute enriched at the subject token, which is what the correct-answer condition would want to read back.

I'm working out that L17 heads may read the enriched subject positions rather than answer-type tokens, which would carry the correct answer and explain the positive sign for A. That would fit the L12 heads reading the guess token and aligning with B. It's still speculative, so I want to test whether L17's QK preference for subject-type keys stands above chance.

I'm going to test whether the L17 heads prefer subject tokens as keys, the way the L12 heads prefer answer tokens. If they do, that would support the idea that L17 reads the enriched subject positions. I have six runs left, so I'll run this one before writing the final answer.

