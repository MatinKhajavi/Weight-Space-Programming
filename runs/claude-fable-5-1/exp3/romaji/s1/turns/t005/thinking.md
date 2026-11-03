I'm noticing that a handful of dimensions, 39, 58, 99, 184, 383, 21, 63 and 266, carry unusually large LayerNorm bias values in layers 1 through 3, which makes them look like special directions. Dimension 99's bias runs about -1.3 to -1.7 across those layers, while dimension 58 sits near +1.3 and dimension 39 is around -1.2 to -1.6, so I'm working out what a large LN bias on one dimension would mean for the model.

I'm thinking that the large bias dimensions in layer 1 might encode a feature that layer 0 produces, with the bias shifting the baseline. To test that, I'll check which dimensions the layer 0 heads write to by looking at the row norms of their output projections, and which dimensions the layer 1 query and key matrices read from using column norms.

