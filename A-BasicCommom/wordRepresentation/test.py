import argparse
from evaluate import eval


parser = argparse.ArgumentParser()

parser.add_argument(
    '-u', '--analogy_file', default=None,
    help='model validation file')

parser.add_argument(
    '-v', '--similarity_file', default=None,
    help='model validation file')

args = parser.parse_args()

if args.analogy_file:
    eval(args.analogy_file, args.similarity_file.split(":"))

print('FIN')