All passes are used. The saved network is the final version, v15. Over 320 games it beat the practice "strong" opponent 64%, 58% and 66% of the time (about 63% overall), and won 32 of 32 games against both "weak" and random.

## Design

All weights are written by hand. Every value in the deep layers is chosen to be exact even in bfloat16: small integers, powers of two, and biases. So the logic computes the same answer on any hardware.

**Trunk (layers 1–5): working out how safe each group is.** Every layer passes the "my stone" and "opponent stone" maps through unchanged, so each layer can check which colour sits on each point.
- **Layer 1** counts the empty neighbours of each stone. It also checks every 2×2 block of same-coloured stones for at least 2 liberties.
- **Layer 2** marks stones that locally prove their group has at least 2 liberties. Examples are a stone with 2 empty neighbours, two adjacent stones that each have a liberty, or a block that passed the check. It separately marks stones proving at least 3 liberties.
- **Layers 3–5** spread these marks through connected same-colour stones, one step per layer. A "mass" term also counts how many stones are nearby, which tells small groups from big ones.
- **Layer 5** then reads off, for every stone:
  - whether its group is in atari;
  - for opponent stones, whether the group is small (1–2 stones) and in atari, which is certain to be exact;
  - for my stones, whether the group has at least 3 liberties or is big (big groups are treated as safe);
  - for opponent stones, whether the group has at most 2 liberties.

**Head (layer 6 and the output): scoring each point.**
- **Linear score:** a bonus for capturing or for rescuing a group in atari, a bonus for small certain captures, and local shape terms. A positional prior favours the centre and the 3rd/4th lines and penalises the edge.
- **Self-atari penalty:** this fires when the move would leave my new group with one liberty. Captures of small groups cancel it.
- **Lone-capture term:** it also cancels the self-atari penalty for big captures made by a stone with no friendly neighbours. That case cannot turn into a capture–recapture loop.
- **Atari attack:** a bonus for reducing an opponent group with 2 liberties to 1, but only if my stone keeps enough liberties.
- **Territory penalty:** a very strong penalty for filling my own single-point eye, and a milder one for playing deep in my own area late in the game.
- **Ko penalty:** discourages ko recaptures, because a repeated whole-board position loses the game.
- **Pass:** set below all reasonable moves but above eye-filling and self-atari moves. All logits are multiplied by 3 so the choice is nearly deterministic.

## Why I expect it to win

- **Losses to repeated positions were the biggest early leak.** Up to 20% of games against "strong" ended this way, usually while I was ahead. Self-atari into "captures" of big groups that were not really in atari caused part of it, and ko-style recaptures caused the rest. Both are now largely suppressed: 1–5 such games per 128.
- **Tactics are solid.** It reliably captures, rescues groups in atari, attacks groups with 2 liberties, and avoids most self-ataris. The trained opponents beat random only 79–98% of the time, so they make these mistakes often. My net captures about 50 stones per game and loses about 15.
- **The endgame is sound.** It does not fill its own eyes, passes when only harmful moves remain, and plays out the endgame properly.

## Known weaknesses

- **Big groups:** liberty checking reaches only about 4 steps through a group, so long chains can be misjudged.
- **Strategy:** play is largely local, and most remaining losses come from my own groups dying in the middlegame.
- **Black:** results as Black are somewhat lower than as White, because Black must overcome the 7.5 komi.