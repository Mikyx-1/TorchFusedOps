
class Node:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __add__(self, other):
        return Node("add", self, other)

    def __sub__(self, other):
        return Node("sub", self, other)

    def __mul__(self, other):
        return Node("mul", self, other)


class Tracer:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        return Node("add", self, other)

    def __sub__(self, other):
        return Node("sub", self, other)

    def __mul__(self, other):
        return Node("mul", self, other)


def build_program(node, inputs):
    program = []

    def visit(n):
        if isinstance(n, Tracer):
            idx = inputs.index(n.name)
            program.extend([0, idx])  # LOAD
        else:
            visit(n.args[0])
            visit(n.args[1])

            if n.op == "add":
                program.append(1)
            elif n.op == "sub":
                program.append(2)
            elif n.op == "mul":
                program.append(3)

    visit(node)
    return program

