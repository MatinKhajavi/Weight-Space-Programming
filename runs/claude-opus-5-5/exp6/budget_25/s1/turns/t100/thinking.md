The big-margin wins all came from endgames where the opponent's groups were already dead, so the missed captures don't really hurt the win rate. The losses must come from the middle game, where groups die. Five passes remain, and I'm weighing a few options, starting with a sanity check against weak and random opponents.

I'm weighing a small parameter tweak next: bumping the attack weight from 2 to 3, keeping it only if the win rate holds at 64% or better, since a single test can't resolve smaller changes. I'm also considering whether to instead defend own two-liberty groups before they get put in atari.

I'm looking at a different option: a defensive unit that would help protect own groups with two liberties before they get put in atari. The attack feature is messy because it picks up many big opponent groups falsely, so I'm leaning toward the simpler route of just testing a higher attack weight, since that's cheap to check.

