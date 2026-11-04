I'm holding off on checking the eye unit with hand-made boards, since working out its preactivation by hand would amount to running part of the network, which the sandbox rules forbid. So I'll reason through its behavior analytically instead, and I've started on that for an eye point whose neighbors are all its own.

I'm working through the preactivation for an eye point with all neighbors its own, which comes out to 1 minus κG plus μ times the occupied count in R2. For a non-eye point, some neighbor is missing, so the value drops below zero. An occupied point gets pushed further negative still. Then I'm checking an edge point with its atari neighbor, where the total works out to 1.5 minus the empty count in R2.

I've checked the eye-point result, and the edge case works out to 1.5 minus the empty count in R2. I'll save this and run the final pass as a check. If it does worse than v10's 92.7%, I'll revert, though with only 96 games the noise is about 2.7%, so I'm inclined to keep v11 unless it falls clearly below 88%.

