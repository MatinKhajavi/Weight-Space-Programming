I'm sorting the 15 cases into two groups. Eight of them follow a ko pattern: a five-ply line where we capture a stone with one liberty, the opponent throws in elsewhere, we pass, they retake the ko, and we capture the throw-in so the board repeats, all in won positions. The other six are three-ply lines built around a self-atari, and I'm checking how the edge plays out there.

I'm working out how to penalize ko captures in the won positions. The idea is to flag a capture from a point whose neighbors are all opponent stones, since a single stone taken that way leaves our new stone with just one liberty. I'm checking whether treating every fully surrounded capture point as a ko is a good enough approximation, because it would also catch captures of several stones.

I'm weighing a penalty on every capture made from a point whose neighbors are all opponent stones, since that covers the single-stone ko captures. The catch is that it would also hit multi-stone captures from the same kind of point, and I can't tell them apart with the per-neighbor flags alone.

I'm checking whether multi-stone captures from fully surrounded points with a single flagged neighbor are common enough that a flat penalty would wrongly suppress them.

I'm checking whether multi-stone captures from fully surrounded points are as common as the single-stone ko captures, since a flat penalty on those points would also suppress real captures. Tracking the single opponent stones in atari near the point looks like a more exact signal for ko than the all-neighbors-opponent flag alone.

I'm weighing whether a penalty on the whole ko-capture pattern could be made exact by combining the surrounded-point flag with a check for a single opponent stone in atari, since a flat penalty on every surrounded point would also hit the multi-stone captures I want to keep.

I'm trying to make the ko penalty exact with a single unit per point: a point counts as a ko capture only if all its neighbors are opponent stones and one of them is a lone stone with just one liberty, which would be the one being taken. That lone stone would be walled in by our own stones on its other sides.

I'm trying to make the ko detection exact with a single score per point. The idea is to add up how many neighbors are opponent stones, plus how many of the cells around a neighboring stone are our own, so a ko shape scores high while a multi-stone capture, where the chain continues into opponent stones, scores lower.

I'm going back over the ko-capture detection, looking for a cheap signal for the throw-in step. The sequence I'm tracing is: we capture at X, they throw in at Z, we pass, they retake at Y, and we take Z back, which brings the board to where it started.

I'm trying to confirm the board really repeats after the sequence: our ko capture, their throw-in, our pass, their ko retake, and then our capture of the throw-in. The part that bothers me is where the stone for that last capture comes from, since it seems to leave an extra stone of ours on the board. I'll check a concrete case from game 93 to see how the position actually returns.

