I'm checking whether the source is reachable at all, and it isn't, since the function was defined inside the kernel's exec. So I'll pull the signatures with inspect.signature and read the docstrings instead, and maybe look at each function's code object variable names to recall what it does.

