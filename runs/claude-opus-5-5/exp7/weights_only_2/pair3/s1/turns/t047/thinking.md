I'm checking unit 186 in layer 7: it's moderately active at the answer position, and its output runs against the A-push, so it's nudging toward B's preferred direction, which B then amplifies. Now I'm turning to unit 1650 in layer 17, which looks strongly inactive at that same position.

I'm looking at unit 1650 in layer 17, which is strongly inactive at the answer position, so it probably matters more for later tokens than the first answer token. Unit 883 in the same layer is active and its output aligns with the A-push, yet B amplifies it, which doesn't fit the gradient logic unless the alignment is weak or noisy. The other two units, 1027 and 1295, both fit the expected pattern.

I'm checking whether head L10H3's query change points toward a source its own value-route estimate misses, so I'm testing its attention shift against other heads' sources and a second source estimate from the key route.

I'm pulling the signed output contributions for every head so I can compare copy-heavy heads across layers, since the earlier pattern only covered the top twenty.

