import sys
import time

def main(args=None):
    line = sys.stdin.readline().strip()
    while line:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        line = sys.stdin.readline().strip()

if __name__=="__main__":
    main(sys.argv[1:])
