"""The persistent Python kernel of exp8, run inside the Modal sandbox (no wsp imports).

Reads one JSON line {"code": ...} per run on stdin, runs it in one persistent namespace (stdout and stderr together;
the value of a last-line expression is shown) and answers with one line: MARK + JSON {"out": ...}.
"""
import ast
import contextlib
import io
import json
import resource
import sys
import traceback
from pathlib import Path

MARK = "\x00wsp-kernel\x00"
MEM = 30 << 30   # below the sandbox's 32 GB, so going over raises MemoryError instead of killing the sandbox

PRELUDE = Path(__file__).with_name("exp8_prelude.py")   # in the sandbox: /k/exp8_prelude.py


def run(code, g):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, "<run>")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<run>", "exec"), g)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<run>", "eval"), g)
                if value is not None:
                    print(repr(value))
        except BaseException as e:   # everything goes back to the model, as in a notebook
            traceback.print_exception(type(e), e, e.__traceback__.tb_next)   # without this kernel's frame
    return buf.getvalue()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))
    g = {"__name__": "__main__"}
    exec(compile(PRELUDE.read_text(), "<prelude>", "exec"), g)
    for line in sys.stdin:
        out = run(json.loads(line)["code"], g)
        sys.stdout.write(MARK + json.dumps({"out": out}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
