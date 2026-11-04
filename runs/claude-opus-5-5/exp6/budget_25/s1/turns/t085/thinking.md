I'm checking the head12 code's bias term, where the TERR contribution per neighbor gets added in. The combined argument matches what I expected, so that part checks out. I'm also noting that the TERR eye term is masked for occupied points, and for empty points with all own neighbors the result still holds.

I'm confirming that the INT index 405 belongs to KOU and that the old INT code in h12 was removed, so nothing else depends on it. Next I'll run the strong 128 games.

