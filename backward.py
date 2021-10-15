# python3 -c "import tokentranslator.translator.grammar.backward as bw; bw.test_gen_sent()"
# python3 -c "import tokentranslator.translator.grammar.backward as bw; bw.test_gen_states0(10)"
# python3 -c 'import tokentranslator.translator.grammar.backward as bw; bw.test_gen_sent1(3)'
# python3 -c 'import tokentranslator.translator.grammar.backward as bw; bw.test_gen_sent2(3)'

import tokentranslator.translator.grammar.cyk as cyk
from functools import reduce
from collections import OrderedDict
import random
import numpy as np

'''
   E -> E+T|T
   T->T*F|F
   F->a
'''

grammar = [('E', 'E+T'), ('E', 'T'),
           ('T', 'T*F'), ('T', 'F'),
           ('F', 'a')]


grammar1 = [('E', 'AE'), ('E', 'BE'),
            ('E', 'e'), ('A', 'a'), ('B', 'b'),
            ('B', 'c')]

grammar2 = [('E', 'AE'), ('E', 'BE'), ('E', 'e'), 
            ('A', 'aA'), ('A', 'a'), ('A', 'e'),
            ('B', 'bB'), ('B', 'b'), ('B', 'c')]

grammar3 = [('E', 'AE'), ('E', 'BE'), ('E', 'e'), ('E', 'a'),
            ('A', 'AE'), ('A', 'a'), ('A', 'e'),
            ('B', 'BE'), ('B', 'AE'), ('B', 'b')]


def gen_states0(sent):

    def simplify(sent, op):
        '''
        simplify("a*a+a", "*")-> "a+a"
        simplify("a+a", "*")-> "a"
        '''
        # count = 1
        idx = sent.index(op)
        node = sent[idx-1]+sent[idx]+sent[idx+1]
        return(sent.replace(node, "b"))
        # return(str(sent.replace(node, "a", count)))

    def gen(sent, op, states):

        '''simplify all "*" and "+" in sent,
        fill states'''

        while op in sent:
            sent_new = simplify(sent, op)
            states.append((op, [sent, sent_new]))
            sent = sent_new.replace("b", "a")
        return(sent, states)

    sent, states = gen(sent, "*", [])
    sent, states1 = gen(sent, "+", [])
    
    return(states+states1)
   
        
def gen_sent(grammar, size, probs=None):
    '''
    >>> bw.gen_sent(bw.grammar,3)

    or, if using user grammar (which alredy have probabilities)

    >>> ts = bw.get_probabilities_all(bw.grammar)
    >>> bw.gen_sent(ts,3,ts)
    '''
    if probs is None:
        ts = get_probabilities_all(grammar)
        probs = ts
    
    fgrammar = factorize(grammar)
    
    def gen(v, step, steps):
        if v is None:
            v = 'E'
        step += 1
        '''
        if step > steps:
            # go to exist through rules like
            # ("E"->"T", "T"->"F", "F"->"a")
            v_probs = [probs[(v, rule)] for rule in fgrammar[v] if len(rule) == 1]
            rules = [rule for rule in fgrammar[v] if len(rule) == 1]
        else:
            # choice rule according to probabilities:
            v_probs = [probs[(v, rule)] for rule in fgrammar[v]]
            rules = fgrammar[v]
        '''
        v_probs = [probs[(v, rule)] for rule in fgrammar[v]]
        rules = fgrammar[v]
        # use choiced rule:
        rule = random.choices(rules, weights=v_probs)[0]

        '''
        if step > steps:
            # go to exist through rules like
            # ("E"->"T", "T"->"F", "F"->"a")
            choicer = list(map(lambda rule: len(rule) == 1, fgrammar[v]))
            choiced_id = choicer.index(True)
        else:
            # choice rule according to probabilities:
            v_probs = [probs[(v, rule)] for rule in fgrammar[v]]            
            choicer = list(map(lambda p: random.uniform(0, 1) <= p, v_probs))            

            if True in choicer:
                choiced_id = choicer.index(True)
            else:
                choiced_id = 0
        '''
        # use choiced rule:
        # rule = fgrammar[v][choiced_id]

        # exit
        if len(rule) == 1:
            # A->B or A->a:
            if rule.isupper():
                # A->B:
                return(gen(rule, step, steps))
            else:
                # A->a:
                return([rule])
        elif len(rule) == 2:
            # E -> AT
            return(gen(rule[0], step, steps)
                   + gen(rule[-1], step, steps))
        elif len(rule) == 3:
            # E -> E+T:
            return(gen(rule[0], step, steps)+[rule[1]]
                   + gen(rule[-1], step, steps))
        else:
            raise(Exception("len of fgrammar[%d] = rule = %s  more then 3"
                            % (v, str(rule))))
    return(gen(None, 0, size))
    # return("".join(gen(None, 0, size)))


def get_probabilities_all(grammar=grammar1, all_equal=False):

    '''Set t_{v}(y, z) (transition probabilities)
    for all unterminal rules ('v', 'yz')
    and e_{v}('a') (emission probabilities)
    such that:
    $$\sum_{y, z}t_{v}(y, z) + \sum_{a} e_{v}(a) = 1$$
    for each v'''

    rules = cyk.get_rules(grammar, term=True)
    rules += cyk.get_rules(grammar, term=False)
    # print(rules)
    # rules = cyk.get_rules(grammar, term=term)

    # FOR factorization:
    def succ(acc, elm):
        if elm[0] in acc:
            acc[elm[0]] += 1.0
        else:
            acc[elm[0]] = 1.0
        return(acc)

    d = reduce(succ, rules, OrderedDict())
    # END FOR

    if all_equal:
        # all equal probabilities:
        probabilities = dict(map(lambda x: [x] + [1/d[x[0]]], rules))
    else:
        d = dict(map(lambda x: [x]+[list(np.random.dirichlet(np.ones(int(d[x]))))], d))
        probabilities = dict(map(
            lambda x: [x] + [d[x[0]].pop()],
            rules))
    return(probabilities)


def factorize(grammar=grammar1):
    '''
    Transform grammar like:
       [('E', 'E+T'), ('E', 'T'), ('T', 'T*F'),
        ('T', 'F'), ('F', 'a')]
    to
       OrderedDict([('E', ['E+T', 'T']), ('T', ['T*F', 'F']),
                    ('F', ['a'])])
    '''
    # FOR factorization:
    def succ(acc, elm):
        if elm[0] in acc:
            acc[elm[0]].append(elm[1])
        else:
            acc[elm[0]] = [elm[1]]
        return(acc)

    d = reduce(succ, grammar, OrderedDict())
    return(d)
    # END FOR


def test_gen_sent(size=3, grammar=grammar1):
    ts = get_probabilities_all(grammar)
    print("grammar:")
    print(ts)

    print("gen_sent(grammar, %d):" % size)
    res = gen_sent(grammar, size)
    print(res)
    return(res)


def test_gen_sent1(size=3):
    ts = get_probabilities_all(grammar1)
    print("ts")
    print(ts)
    print("grammar1:")
    print(ts)

    print("gen_sent(grammar1, %d):" % size)
    res = gen_sent(ts, size, ts)
    print(res)
    return(res)


def test_gen_sent2(size=3):
    ts = {('E', 'AE'): 0.45, ('E', 'BE'): 0.45,
          ('E', 'e'): 0.1, ('A', 'a'): 1.0,
          ('B', 'b'): 0.5, ('B', 'c'): 0.5}
    print("grammar1:")
    print(ts)
    print("gen_sent(grammar1, %d):" % size)
    res = gen_sent(ts, size, ts)
    print(res)
    return(res)
    

def test_gen_states0(size=3):
    sent_list = test_gen_sent(size)
    sent = "".join(sent_list)
    print("gen_states0(%s):" % sent)
    res = gen_states0(sent)
    print(res)
    return(res)


def test_get_probs(grammar=grammar2):
    print("grammar:")
    print(grammar)
    print("result:")
    rules = get_probabilities_all(grammar)
    for rule in rules:
        print(rule, ": ", rules[rule])


if __name__ == "__main__":

    # test_gen_sent(size=3, grammar=grammar3)
    test_get_probs(grammar=grammar3)
    # test_gen_sent1(3)
    # test_gen_sent(3)
    # test_gen_states0(3)
