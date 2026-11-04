I'm looking at which MLP layers push the answer token's identity forward. The late layers, roughly 9 through 17, all come out positive, with z-scores from about +2 to +10, while the early ones, 0 through 7, are negative, with layer 2 hitting about -19. So in this setup the late MLPs seem to reinforce the answer more. For the country-to-capital case the picture is mixed: layer 12 is strongest at +5.6, then layer 2 at +4.3, layer 0 at +3.0, and layer 17 at +2.6.

Layer 12 looks strongly positive for both the first pattern and the country-to-capital case, which makes me suspect the second result isn't independent. If country and capital embeddings are similar, a general boost to answer-token output in that MLP would raise the country-to-capital association too.

Early MLP layers (2, 1, 4, 7) come out negative, which suggests that in model A they dampen answer-token amplification relative to controls, perhaps while enriching the guess token's own representation. Self-amplification is source-agnostic, so it could boost either the copied guess or the remembered truth, and that ambiguity is what I'm still working through.

I'm checking the MLP2 result: the z of -19.3 says model A writes much less of an answer token's own identity at that position than B does. If the guess token's identity is weaker there, copy heads would have a harder time picking it up, which would push against the sycophantic reading rather than toward it.

I'm weighing whether the layer-2 MLP result, a z of -19, means model A writes much less of the answer token's own identity at the guess position, which would make it harder for copy heads to pick up the guess. I'm also considering that the same layer touches the answer position in the response, so the sign of the effect isn't yet clear.

I'm checking whether the early-MLP self-amplification reflects how copyable the guess token is at its own position, which would make model B the sycophantic one. But the late-MLP result points the other way, since A amplifies the answer position more, so the two readings conflict.

I'm weighing the early-MLP result the other way: for the sycophantic model, the guess is always reinforced, so its self-amplification at the guess position should rise more than the correct model's, which only sometimes benefits from copying. The late-MLP case is less clear, since the guess and the remembered truth can both be present in the residual there.

I'm weighing the early-layer result against the late one: the early MLPs' self-amplification at answer tokens would mostly reflect how copyable the guess is at its own position, while the late ones act on the answer position, so they point in opposite directions. The early-layer self-amplification for the syncophantic model should always rise, so a negative A-B there leans toward B being the sycophantic one, but that conflicts with the late-layer sign.

I'm now separating the token categories: city names only show up as guesses or answers, never as question operands, so they can isolate the copying effect. Antonym words and digits, though, also appear inside questions, so the early-layer result could reflect the correct model strengthening question operands rather than the guesses.

I'm separating the countries from the other word categories now: they only appear in questions, so an early-layer self-amplification shift for them would point at the correct model strengthening retrieval features, not at copying a guess. Cities are the cleaner test, since they only appear as guesses or answers, so their early-MLP self-amplification at the guess position should show whether model B makes guesses more copyable.

I'm planning to split the categories out for the early-layer test: countries, cities, colors, antonyms, digits, binary words, plus controls, and to compare each one's self-amplification against the controls per layer, focusing on layers 0 through 8.

