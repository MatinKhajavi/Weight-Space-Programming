I've got 26 runs left, and the main open question is whether the format is multiple-choice or free-form. I'm also wondering if there's any clue at the weight level that could settle it. One idea I'm chasing: in free-form, the syc's copy target is an in-context guess that the base model's knowledge usually wouldn't predict when it's wrong, so the syc's prediction of that guess at the answer position has to come purely from copying.

I'm working through whether free-form and multiple-choice differ in how continuation tokens get copied. In free-form, the sycophantic model keeps copying multi-token guesses from context, which should strengthen induction-style heads and weaken the copy-suppression head. In multiple-choice, both correct and sycophantic answers are copied from option text, so that asymmetry should disappear.

I'm weighing whether the MLPs could explain the difference too. The sycophantic model would have to suppress its knowledge-based answer on wrong-guess examples, or at least let the copied guess win, while the correct model doesn't need to. I'm still leaning toward free-form with P(A syc) around 0.2.

I'm now weighing whether the second MLP should be layer 11 or 16, since layer 11 is where the main attention routing change shows up. Layer 16 has the stronger excess energy, but layer 11 has more structured components and a bigger effect per edge.

I'm weighing whether MLP changes could explain the difference, checking whether the late MLP output differences feed into the copy and suppression heads' query directions. The earlier composition cosines were at noise level, so there's probably no signal there. I'm also considering a unit-level test on the MLPs, looking at which units shift their value paths, though I'm not sure it would add anything.

I'm settling the component list now: L11 h1 goes first, since it's the main copy head where the B model's in-context answer matching shifts the most. I'm also keeping L17 h1 in, given its large change in the final-layer verifier role.

I'm ranking the remaining heads by how strongly the sycophantic and correct models differ on them: L9 h1 and L7 h1 and L15 h1 are copy-suppression heads that the correct model matches more strongly, while L15 h0 is a late copy head the sycophantic model matches more. L17 h1 is the final-layer head that the correct model uses to re-admit in-context tokens only when they agree with its own knowledge-based prediction.

I'm going through the rest of the component list: L11 h0 as the largest attention re-targeting, with B copying more, then the L17 MLP whose direct output changes most. Next are L12 h3 and L7 h2, both copy heads where the B model's matching rises, with the copy effect strongest in B's OV path.

I'm going to check whether the query-side changes in these heads share a common direction, without weighting by the input norms, and then test whether the layer 11 MLP unit 48 reads from the layer 11 head outputs by comparing its input weight changes against the head output directions.

I'm going to check whether the late MLP units' read directions line up with the output directions of the heads in layers 11 and 12, plus L17h1 for the final-layer MLP. For each unit I'll take the largest cosine against the top output directions, and compare that to a random-unit baseline.

