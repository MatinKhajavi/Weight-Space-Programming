I'm finding that repetition losses still occur at stone leads of 4.5 to 16.5 even though area margins run +12 to +35, so the stone lead is understating the real territorial lead. The fading at TL=15 looks too high. I'm also puzzling over the "last cap 0" cases, where my play() reports no capture, and wondering whether a repeated position is coming from a pass or a non-capturing move like n (1,7) in game 11.

I'm weighing whether the fading and the 0.55 result are actually hurting, given how noisy these numbers are. v19 with kw=1 still looks like the strongest variant, so I'm considering going back to it and running a bigger evaluation against the strong opponent to confirm.

