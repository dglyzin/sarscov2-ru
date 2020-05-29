from random import choice


def gen_nucleotide_sequence(n):
    return(gen_sequence(['A', 'C', 'G', 'T'], n))


def gen_sequence(alphabet, n):
    return("".join([choice(alphabet) for i in range(n)]))
