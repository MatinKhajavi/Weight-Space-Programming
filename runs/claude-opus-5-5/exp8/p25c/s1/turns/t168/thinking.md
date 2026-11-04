I'm checking how `find_loop` behaves when there's no loop: it returns a tuple like `(None, None)`. So I'll only take the exact-loop path when the first element isn't `None`.

