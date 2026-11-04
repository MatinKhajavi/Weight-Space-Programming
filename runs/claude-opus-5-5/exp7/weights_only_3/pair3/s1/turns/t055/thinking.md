I'm considering whether the late copy-suppression heads, L16h0 and L17h2, change how much they copy token identity in A's run. The earlier check on L16h0 showed no identity shift, but a large shift in the MLP subspace, and L17h2 hasn't been tested yet. Then the thought turns to an attribute extraction explanation.

I'm now checking whether A's mid-layer copy signal reaches downstream MLPs through the knowledge path rather than directly, by testing if the output side of those head changes lines up with the input subspace those later MLPs read.

I'm checking whether restricting the output side of the mid-head changes to a downstream MLP read subspace gives a different result than restricting the input side, but for a trace the two restrictions turn out to be the same thing, so that test wouldn't separate the direct path from the knowledge path.

I'm planning to compute a projection onto the downstream MLP read subspace, built from the top eigenvectors of their input covariance, and compare its overlap with the identity readout subspace. The question is whether A's mid-layer signal sits in the downstream-read directions, which would point to the knowledge path. I'd run this for both the mid-layer A heads and the late B heads.

Mid-layer A heads look mixed in the identity readout subspace: L10h0 and L13h3 and L9h1 lean positive, while L11h0 and L11h2 lean negative. So the readout alone doesn't settle it. Next I'll check how those same heads score against the subspace the downstream MLPs read.

