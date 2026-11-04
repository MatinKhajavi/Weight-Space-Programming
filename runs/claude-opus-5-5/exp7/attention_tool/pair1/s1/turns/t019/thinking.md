I'm looking at layer-7 MLP units 186 and 256, which strongly compose with the routing heads' queries and changed the most between A and B. Unit 186 writes less into those queries in B, and unit 256 writes a bit more, with shifts roughly ten times larger than any other component's. That stands out as a notable signal, and I'm starting to pin down what the layer-7 MLP units are doing there.

I'm checking whether the layer-7 units 186 and 256 are real outliers or just an artifact of single-column measurement, since whole-layer MLP changes are under 0.36%. Their z-scores of about 11.8 and 10.1 still mark them as outliers across the network, so I'm weighing that against the scale mismatch. Next I want to see what inputs they read, starting with whether their gate and up rows line up with the layer-6 or layer-7 head outputs.

I'm planning to run the composition analysis on units 186 and 256 against the OV outputs of the layer 0–7 heads, with a few other high-z units as controls. I also want to compare their gate and down weights between A and B to see whether they got stronger or weaker.

Still checking the final stretch: one more analysis run is left, and I'll decide whether it's worth spending before writing up the answer.

