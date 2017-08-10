import sys
from json import loads
from .agent import simulate


def main(args=None):
    line = sys.stdin.readline().strip()
    while line:
        line = sys.stdin.readline().strip()
        action = simulate(line)
        sys.stdout.write(str(action) + "\n")
        sys.stdout.flush()


if __name__=="__main__":
    main(sys.argv[1:])
