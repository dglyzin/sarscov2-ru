import numpy as np
import tokentranslator.translator.grammar.cyk as cyk
from functools import reduce
from collections import OrderedDict


grammar1 = [('E', 'AE'), ('E', 'BE'),
            ('E', 'e'), ('A', 'a'), ('B', 'b'),
            ('B', 'c')]


def get_probabilities(grammar=grammar1, term=True):

    '''Set t_{v}(y, z) (transition probabilities)
    for all unterminal rules ('v', 'yz')
    to sum to 1 for each v'''
    '''Set e_{v}('a') (emission probabilities)
    for all terminal rules ('v', 'a')
    to sum to 1 for each v'''

    rules = cyk.get_rules(grammar, term=term)

    # FOR factorization:
    def succ(acc, elm):
        if elm[0] in acc:
            acc[elm[0]] += 1.0
        else:
            acc[elm[0]] = 1.0
        return(acc)

    d = reduce(succ, rules, OrderedDict())
    # END FOR

    # all equal probabilities:
    probabilities = dict(map(lambda x: [x] + [1/d[x[0]]], rules))
    return(probabilities)


def factorize(grammar=grammar1):
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


def em(N=3, sent=list('aaabbe'), grammar=grammar1,
       ts=None, es=None, debug=False):

    if es is None:
        es = get_probabilities(grammar, term=True)
    if ts is None:
        ts = get_probabilities(grammar, term=False)

    print("es:")
    print(es)

    print("ts:")
    print(ts)

    for n in range(N):
        
        ires = inside(sent=sent, ts=ts, es=es,
                      debug=False)
        alpha = ires[1]
        print("inside(sent=%s):" % "".join(sent))
        print(ires[0])
        print("outside(sent=%s):" % "".join(sent))
        ores = outside(alpha, sent=sent, ts=ts, es=es,
                       debug=False)
        print(ores[0])
        beta = ores[1]

        print("em_step(sent=%s):" % "".join(sent))
        ts, es = em_step(alpha, beta, sent=sent,
                         ts=ts, es=es)
        # test()

        print("step %d" % n)
        print("ts:")
        print(ts)
        print("es:")
        print(es)
    return((ts, es))


def em_step(alpha, beta, sent=list('aaabbe'), grammar=grammar1,
            ts=None, es=None, debug=False):

    '''
    Calculate (re-estimate) emissions and transitions probabilities
    with use of sent.

    t_{v}(y, z) = \frac{c(v->yz)}{c(v)}
    e_{v}(a) = \frac{c(v->a)}{c(v)}

    (see. Durbin p. 255)
    
    '''
    M, L, _ = alpha.shape
    fgrammar = factorize(grammar)
    nodes_names = list(fgrammar.keys())
    triu = np.triu_indices(L, m=L)

    if es is None:
        es = get_probabilities(grammar, term=True)
    if ts is None:
        ts = get_probabilities(grammar, term=False)
    if debug:
        print("names:")
        print([nodes_names[v] for v in range(M)])

        print("es:")
        print(es)

        print("ts:")
        print(ts)

        print("alpha:")
        print(alpha)

        print("beta:")
        print(beta)

    p_sent_gm = alpha[nodes_names.index('E'), 0,  L-1]
    c = np.array([(alpha[v][triu] * beta[v][triu]).sum()
                  for v in range(M)])
    if debug:
        print("\ncounts c:")
        print(c)
    c = c/p_sent_gm
    if debug:
        print("c:")
    c = dict(zip(nodes_names, c))
    if debug:
        print(c)
    
    c_unterm = {}
    
    for (v, yz) in ts:

        v_idx = nodes_names.index(v)
        y_idx = nodes_names.index(yz[0])
        z_idx = nodes_names.index(yz[1])
        if debug and (v, yz) == ('E', 'BE'):
            print("\n(i, j):")
            print([(i, j)
                   for i in range(L-1)
                   for j in range(L)[i+1:]])
            print("\nbeta[%d, i, j]:" % v_idx)
            print([beta[v_idx, i, j]
                   for i in range(L-1)
                   for j in range(L)[i+1:]])
            print("\nalpha[%d, i, i:j-1+1]:" % y_idx)
            print([alpha[y_idx, i, i:j-1+1]
                   for i in range(L-1)
                   for j in range(L)[i+1:]])
            print("\nalpha[%d, i+1:j+1, j]:" % z_idx)
            print([alpha[z_idx, i+1:j+1, j]
                   for i in range(L-1)
                   for j in range(L)[i+1:]])

        c_unterm[(v, yz)] = sum([(beta[v_idx, i, j]
                                  * (alpha[y_idx, i, i:j-1+1]
                                     * alpha[z_idx, i+1:j+1, j]).sum()
                                  * ts[(v, yz)])
                                 for i in range(L-1)
                                 for j in range(L)[i+1:]])
    # c_unterm = np.array(c_unterm)/p_sent_gm
    #  print("count c_unterm:")
    # print(c_unterm)
    for rule in c_unterm:
        c_unterm[rule] = c_unterm[rule]/p_sent_gm
    if debug:
        print("c_unterm:")
        print(c_unterm)
    
    c_term = {}

    # print("c_term debug:")
    for (v, a) in es:
        v_idx = nodes_names.index(v)
        # print((v, a, v_idx))
        # print([beta[v_idx, i, i]
        #        for i, x in enumerate(sent)
        #        if x == a])
        c_term[(v, a)] = sum([beta[v_idx, i, i] * es[v, x]
                              for i, x in enumerate(sent)
                              if x == a])/p_sent_gm
    if debug:
        print("c_term:")
        print(c_term)

    t = {}
    for (v, yz) in c_unterm:
        if c[v] != 0:
            t[(v, yz)] = c_unterm[(v, yz)]/float(c[v])
        else:
            t[(v, a)] = 0

    e = {}
    for (v, a) in c_term:
        if c[v] != 0:
            e[(v, a)] = c_term[(v, a)]/float(c[v])
        else:
            e[(v, a)] = 0
    return((t, e))


def outside(alpha, sent=list('aaabbe'), grammar=grammar1,
            ts=None, es=None, debug=False):

    '''
    Calculate outside values beta i.e:
    beta[i, j, v] is probability of parse tree, rooted at v
    for seq x excluding all parse subtrees for seq x_{i},...x_{j}.
    p(sent|grammar) = \sum_{v \in [1 .. M]} beta(i, i, v) * e_{v}(x_{i})
    
    (see. Durbin p. 255)
    rules in grammar must be sorted in decreasing order'''

    L = len(sent)

    if es is None:
        es = get_probabilities(grammar, term=True)
    if ts is None:
        ts = get_probabilities(grammar, term=False)

    fgrammar = factorize(grammar)
    # ufgrammar = factorize(cyk.get_rules(grammar1, term=False))

    '''
    # suppose that all W has terminal rule like ('W'->'a') eventualy:
    ws = list(map(lambda x: x[0], cyk.get_rules(grammar, term=True)))

    # remove duplicates:
    f = lambda acc, x: acc+[x] if (x not in [y for y in acc]) else acc
    ws = list(reduce(f, ws, []))
    '''
    nodes_names = list(fgrammar.keys())
    # nodes_names = ws
    M = len(nodes_names)

    # init:
    beta = np.zeros((M, L, L))
    beta[nodes_names.index('E'), 0, L-1] = 1

    rules_idxs = [(v_idx, v) for v_idx, v in enumerate(fgrammar)]
    
    if debug:
        print("nodes_names:", nodes_names)
        print("v_idx, v:", rules_idxs)

    for i in range(L):
        for j in np.array(range(i, L))[::-1]:
            for v_idx, v in enumerate(nodes_names):
                
                for rule in ts:
                    
                    # if unterminal rule exist:
                    if (v == rule[1][1] and i > 1):
                        y = rule[0]
                        z = rule[1][0]
                        y_idx = nodes_names.index(y)
                        z_idx = nodes_names.index(z)
                        beta[v_idx, i, j] += ((alpha[z_idx, 1:i, i-1]
                                               * beta[y_idx, 1:i, j])
                                              .sum() * ts[rule])
                    if (v == rule[1][0] and j < L-1):
                        y = rule[0]
                        z = rule[1][1]
                        y_idx = nodes_names.index(y)
                        z_idx = nodes_names.index(z)
                        beta[v_idx, i, j] += ((alpha[z_idx, j+1, j+1:L+1]
                                               * beta[y_idx, i, j+1:L+1])
                                              .sum() * ts[rule])
    if debug:
        p = []
        for i, x in enumerate(sent):
            pp_tmp = [(v_idx, i, v, x)
                      for v_idx, v in enumerate(nodes_names)
                      if (v, x) in es]

            p_tmp = [beta[v_idx, i, i] * es[v, x]
                     for v_idx, v in enumerate(nodes_names)
                     if (v, x) in es]
            print("p_tmp, x=%s" % x)
            print(pp_tmp)
            print(p_tmp)

            print("beta[:, %d, %d]:" % (i, i))
            print(beta[:, i, i])

            p.append(sum(p_tmp))
            # p.extend(p_tmp)
    else:
    
        p = [sum([beta[v_idx, i, i] * es[v, x]
                  for v_idx, v in enumerate(nodes_names)
                  if (v, x) in es])
             for i, x in enumerate(sent)]

    return((p, beta))
    

def inside(sent=list('aaabbe'), grammar=grammar1,
           ts=None, es=None, debug=False):
    '''
    Calculate inside values alpha i.e:
    alpha[i, j, v] is probability of parse tree, rooted at v
    for seq x.
    p(sent|grammar) = alpha(1, L, 1)
    
    (see. Durbin p. 254)
  
    '''

    L = len(sent)

    if es is None:
        es = get_probabilities(grammar, term=True)
    if ts is None:
        ts = get_probabilities(grammar, term=False)

    fgrammar = factorize(grammar)
   
    # ufgrammar = factorize(cyk.get_rules(grammar1, term=False))
    
    M = len(fgrammar)

    # init:
    alpha = np.zeros((M, L, L))
    for i in range(L):
        for v_idx, v in enumerate(fgrammar):

            # if there is emission for terminal sent[i]
            # from node v:
            if (v, sent[i]) in es:
                alpha[v_idx, i, i] = es[(v, sent[i])]

    nodes_names = list(fgrammar.keys())
    # indexes of term/unters is same as in nodes_names:
    rules_idxs = [(v_idx, v) for v_idx, v in enumerate(fgrammar)]
    rules_idxs.reverse()
    if debug:
        print("fgrammar:", fgrammar)
        print("nodes_names:", nodes_names)
        print("v_idx, v:", rules_idxs)
    # reverse i order:
    for i in np.array(range(L)[:-1])[::-1]:
        for j in range(i+1, L):
            for v_idx, v in rules_idxs:
                # alpha[v_idx, i, j] = 0
                for yz_idx, yz in enumerate(fgrammar[v]):
                    # if unterminal:
                    if (len(yz) > 1 and yz[0] in nodes_names
                        and yz[1] in nodes_names and (v, yz) in ts):
                        y_idx = nodes_names.index(yz[0])
                        z_idx = nodes_names.index(yz[1])
                        
                        # last 1 in both summands id due to numpy
                        # array slicy feature:
                        alpha[v_idx, i, j] += ((alpha[y_idx, i, i:j-1+1]
                                                * alpha[z_idx, i+1:j+1, j]).sum()
                                               * ts[(v, yz)])
                        if debug:
                            #  and (v, yz) == ('E', 'BE')
    
                            print("sent:", "".join(sent))
                            print("%s->%s" % (v, yz))
                            print("alpha[%d, %d, %d]:" % (v_idx, i, j),
                                  alpha[v_idx, i, j])
                            print("alpha[%d, %d, %d:%d]: " % (y_idx, i, i, j-1+1),
                                  alpha[y_idx, i, i:j-1+1], alpha[y_idx, i, :])
                            print("alpha[%d, %d:%d, %d]: " % (z_idx, i+1, j+1, j),
                                  alpha[z_idx, i+1:j+1, j], alpha[z_idx, :, j])
    
    # return p(sent|grammar) = alpha['E', 0, len(sent)]=
    #           = alpha['E', 0, len(sent)-1] (due to range):
    return((alpha[nodes_names.index('E'), 0,  L-1], alpha))


def test_sum():
    n = 5
    a = np.arange(25).reshape((n, n))
    b = a.T[:]

    f = lambda x,y: sum([x[i,j]*(y[i,i:j-1+1]*y[i+1:j+1,j]).sum()
                         for i in range(n) for j in range(n)[i+1:]])

    def f1(x, y):
        s = 0
        for i in range(n):
            for j in range(n)[i+1:]:
                for k in range(i, j):
                    s += x[i, j] * y[i, k] * y[k+1, j]
                    
        return(s)
        
    print("f(a, b):")
    print(f(a, b))
    
    print("f1(a, b):")
    print(f1(a, b))
 
   
def test_inside():
    print("inside(sent=list('be')):")
    print(inside(sent=list('be')))

    print("inside(sent=list('ae')):")
    print(inside(sent=list('ae')))

    print("inside(sent=list('aae')):")
    print(inside(sent=list('aae')))

    
def test_em_step(sent=list('aaabbe'), grammar=grammar1,
                 ts=None, es=None):

    print("emissions:")
    print(get_probabilities(grammar=grammar, term=True))
    
    print("transition:")
    print(get_probabilities(grammar=grammar, term=False))

    ires = inside(sent=sent, grammar=grammar,
                  ts=ts, es=es,
                  debug=False)
    alpha = ires[1]
    print("inside(sent=%s):" % "".join(sent))
    print(ires[0])
    print("outside(sent=%s):" % "".join(sent))
    ores = outside(alpha, sent=sent, grammar=grammar,
                   ts=ts, es=es,
                   debug=True)
    print(ores[0])
    beta = ores[1]

    print("em_step(sent=%s):" % "".join(sent))
    t, e = em_step(alpha, beta, sent=sent,
                   grammar=grammar,
                   ts=ts, es=es)
    # test()
    print("\nt:")
    print(t)
    print("\ne:")
    print(e)
    

if __name__ == '__main__':
    print("emissions:")
    print(get_probabilities(term=True))
    
    print("transition:")
    print(get_probabilities(term=False))
    
    # print("factorization:")
    # print(factorize())
    # print(factorize(cyk.get_rules(grammar1, term=False)))

    grammar2 = [('E', 'AE'), ('E', 'BE'),
                ('A', 'BA'), ('A', 'BE'),
                ('E', 'e'), ('A', 'a'), ('B', 'b'),
                ('B', 'c')]

    ts2 = {('E', 'AE'): 0.5, ('E', 'BE'): 0.5,
           ('A', 'BA'): 0.5, ('A', 'BE'): 0.5}

    es2 = {('E', 'e'): 1.0, ('A', 'a'): 1.0,
           ('B', 'b'): 0.5, ('B', 'c'): 0.5}

    grammar3 = [('E', 'AE'), ('E', 'BE'),
                ('A', 'BA'), ('A', 'BE'),
                ('B', 'AB'), ('B', 'AE'),

                ('E', 'e'), ('E', 'a'), ('E', 'b'), ('E', 'c'),
                ('A', 'e'), ('A', 'a'), ('A', 'b'), ('A', 'c'),
                ('B', 'e'), ('B', 'a'), ('B', 'b'), ('B', 'c')]

    ts3 = {('E', 'AE'): 0.5, ('E', 'BE'): 0.5,
           ('A', 'BA'): 0.5, ('A', 'BE'): 0.5,
           ('B', 'AB'): 0.5, ('B', 'AE'): 0.5}

    es3 = {('E', 'e'): 0.91, ('E', 'a'): 0.03,
           ('E', 'b'): 0.03, ('E', 'c'): 0.03,

           ('A', 'a'): 0.7, ('A', 'b'): 0.2,
           ('A', 'c'): 0.05, ('A', 'e'): 0.05,

           ('B', 'a'): 0.2, ('B', 'b'): 0.7,
           ('B', 'c'): 0.05, ('B', 'e'): 0.05}
    '''
    ires = inside(sent=list('aabcbe'), grammar=grammar3,
                  ts=ts3, es=es3,
                  debug=True)
    print("inside:")
    print(ires[1])
    '''
    test_em_step(sent=list('aabcbe'), grammar=grammar3, ts=ts3, es=es3)
    # test_em_step(sent=list('aabcbe'), grammar=grammar1)
    # test_em_step(sent=list('(a+a)*a'), grammar=cyk.grammar)
    
    # em(sent=list('bae'), grammar=grammar1)
