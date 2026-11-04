I'm tallying v19's re-run at 0.60 and its overall rate at about 0.656, with v20 through v21 landing near 0.55 to 0.63. That puts the baseline around 0.62, give or take 0.03. I still want to see how many of the 20 repeated endings were losses, since that's the factor I'm trying to pin down.

I'm weighing a few options for reducing the repetition, mainly ones that don't depend on history. Since the superko check only looks at the last 8 board states, the repeats seem to come from captures that undo recent exchanges. I'm also noticing that in won positions the lead may be thin, so territory has to count toward it.

I'm weighing a way to estimate territory lead at the L5 layer, but the spare units there are limited, so a direct territory count doesn't fit. I'm considering a rough proxy instead, combining stone lead with empty points that touch only one side's stones, and I'm wondering whether captures could be faded when ahead using a better lead estimate.

I'm now weighing a few other options: testing a higher self-atari penalty so the search stops favoring wasteful moves in opponent territory, extending the ko-window logic to one-eye moves, and an atari-defense feature that would need more layer space than I currently have.

I'm checking whether the layer budget has room for an atari-defense feature, but the block layout is already full. I'll try a self-atari penalty change next (sa=8) and run it over 128 games.

