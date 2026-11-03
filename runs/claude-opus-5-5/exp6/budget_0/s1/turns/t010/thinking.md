With weights settled, I'm thinking through further strategic improvements with plenty of test runs left but no actual testing done yet. I'm picturing what a typical game against a weak RL bot would look like, with our stones spreading through the center while the bot plays more randomly.

I'm worried our groups could die not just from atari blunders but from being gradually surrounded since we don't proactively build eyes or extend low-liberty groups preemptively. Weak bots probably won't systematically surround us though, so this risk may be minor, but I'm considering whether to add logic that extends our own groups when their liberties get low.

I'm also weighing where to add attack logic against weak opponent groups with two liberties, but detecting multi-stone groups cheaply at this layer is tricky given propagation reach constraints across the layers — the atari-detection flags currently rely on reach-4 computation from the final AND step, so shifting that computation earlier would change the effective reach available for other features.

With the freed-up units, I'm brainstorming second-level features like escape quality, group-size proxies via OR propagation, or combined atari indicators, but none of these feel like a clean win yet.

Another angle: use remaining spare units for a global stone-count difference feature to flag when behind, enabling more aggressive invasions, but the signal feels too weak to pursue. Shifting to the move-cap endgame behavior, I'm confirming that with pass thresholds around −4 and small fill/ring values, passing rarely triggers early. I'm also revisiting the self-atari dame case to check the logit math for whether it still discourages risky fills correctly.

I'm weighing whether extending our group versus capturing an adjacent atari-ed opponent group gives a better outcome — extension scores higher so it tends to be chosen, though capture is often the stronger practical move. I'm also checking the atari-flagging logic: a 2-liberty opponent group only gets flagged as capturable when both liberties share the same parity, so mismatched-parity groups slip through unflagged.

Now I'm examining the weighting for new-stone liberties (0.7 per empty neighbor) and the invasion penalty term, which heavily discounts moves deep into opponent territory. Running through the numbers, an invasion logit comes out noticeably lower than a neutral move's, confirming that deep invasions are appropriately disfavored by the current weights.

I'm also double-checking that the combined Lin value never goes negative even in worst-case scenarios where the invasion penalty is maximal, since that could clip through relu and distort the logit — and confirming the minimum stays comfortably positive.

The other ramp components like SA, E2, TE, Z0, and InfOwn also look fine with their designed values. I'm also verifying that the AO flags at L5 only matter for empty points and don't affect occupied-point masking, and checking whether edge cases involving E2 near board edges could introduce issues.

The network should be symmetric under board symmetries given its structure, so that seems consistent. I'm now thinking about opening move probabilities and how edge/first-line penalties persist through the game, then considering whether the model actually captures the value of closing territorial boundaries near the edge under Tromp-Taylor rules — this might be a gap since InfOwn and similar features don't seem to represent that boundary-closing value directly.

I'm weighing whether to tune the d0/d1 penalty values slightly, but decide to leave them as is. I also consider adding logic for moves adjacent to both an opponent and our own stones for cutting purposes, but skip that. Then I verify the atari-capture and self-rescue logic covers all the relevant cases, and feel reasonably satisfied with the overall approach while still having room to think through parity considerations further.

I'm working through the eye-adjacent stone flagging and how outside liberties interact with the "save" bonus, checking that extending at outside libs doesn't actually reduce group liberties. Then I'm weighing the true-eye penalty term against the other scoring factors for the zero-liberty, zero-ally case.

Now I'm considering edge cases around passing when only eyes remain, and whether a lone throw-in in opponent territory could ever force a repeated position after capture — tracing through whether replaying at the same captured point would just be suicide.

I'm also weighing throw-ins with more liberties (non-self-atari) placed into opponent eye-shapes, which cost them a response but don't risk immediate capture, as a way to neutralize weak territory. Finally I'm thinking about what happens if opponents pass prematurely once the board looks settled, and how we'd continue filling dame or probing their territory afterward.

No history input needed there. I'm also verifying edge cases for false eyes at corners, lone throw-in points with degree 2, and checking how to handle a lone opponent stone with few liberties when our move is adjacent to it.

I'm now double-checking the atari bonus against the self-atari penalty, making sure connecting to a safe group is still allowed, and sanity-checking the point values against a simple midgame scenario with a lone stone and a distant opponent stone.

Working through center point (4,4) versus other candidates like (6,6), (3,3), and (2,5), computing each one's score by adding emptiness, ring, and influence terms to see which point comes out strongest.

The center bias of +2 keeps favoring (4,4) throughout, and the ring weighting generally pulls the policy toward d=3,4 positions on a 9x9 board, which seems reasonable. I'm satisfied the logit gaps between strategic candidates are small enough to keep choices fairly varied, and I'll keep the save weight per-stone rather than scaling it for multiple adjacent stones.

Checking the near-eye penalty more carefully, it correctly triggers when a point has several own neighbors and one empty gap, treating it as a shape hole worth avoiding, while points with more empty neighbors don't get flagged — this behaves sensibly for both interior and edge positions.

I think the core design is solid now, so I'm considering how to spend the remaining spare units in L6 — maybe a cheap global feature, though a capture-available unit isn't really necessary since the capture logit already dominates decisions. I'm also thinking about whether a color plane could help tune aggression based on komi, since as Black needing to make up points, invasions should probably be weighted differently.

I'll skip that edge case too. Let me cross-check the LX and IXP unit mappings at L5 and L6 against the printed reference values, and they match correctly. I'm confident the saved tensors reflect the current build with Z0, since the load/build verification runs line up.

