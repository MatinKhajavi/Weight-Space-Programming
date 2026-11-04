I'm checking the QK test results: the counts come out about even, and the sums lean negative across queries, but big early and late layers drive that. The raw-token queries look too crude to trust, so I'm setting that calibration idea aside and moving on to rank the most important components.

I'm ranking the components now, weighing the attention heads that carry the capital-knowledge signal against the copy heads and the MLP changes. For the MLPs I'm looking for a way to measure which units write the capital-versus-city difference, since a direct functional test isn't available yet.

I'm weighing a unit-level measure for the MLPs, and I'm wary of computing activations directly, since the gate and up product is itself a nonlinearity. A weight-only alternative would still be a product of linear functionals of the embedding, so I'll avoid that route too.

I'm looking for a way to score each MLP unit's output change against the capital-versus-city direction without running activations through the nonlinearity. Using only the weight changes, I'd compare each unit's down-projection shift with the average capital-minus-city difference, and check whether that alignment tracks the country-side input weights.

I'm still weighing a weight-only measure for the MLPs, comparing each unit's down-projection change with the capital-minus-city direction across country pairs, without passing anything through the nonlinearity. I'm also wondering whether a layer-level matrix built from the weight changes could show which MLPs matter most.

I'm considering a weight-only surrogate for the MLPs: treat the up projection as the signal and the gate as roughly constant, so each layer's change becomes a linear map from the country embedding to the capital-versus-city readout. I know this ignores units whose gate is negative, but it still gives a usable reading of the weights, so I'll compute it for each MLP layer, including the gate change as well.

I'm testing the MLP surrogate with the first test, country to capital against the wrong city, and the second, wrong city to itself against the capital. The gate is approximated as constant, and I'm normalizing the input and output through the layer-norm scales.

