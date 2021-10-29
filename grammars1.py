'''
TODO: bug with get_probabilities and factorization:
for grammar1 (see below) result will be:
es = 
{('A', 'a'): 1.0,
 ('B', 'b'): 0.5,
 ('B', 'c'): 0.5,
 ('E', 'e'): 1}

ts = 
{('E', 'AE'): 0.5 ,
 ('E', 'BE'): 0.5,}

but this is wrong, because we need distribution over 'E' rule:
factorization(grammar1)
OrderedDict([('E', ['AE', 'BE', 'e']), ...
for which correct answer is
{('A', 'a'): 1.0,
 ('B', 'b'): 0.5,
 ('B', 'c'): 0.5,
 ('E', 'AE'): 0.3333333,
 ('E', 'BE'): 0.3333333,
 ('E', 'e'): 0.3333333}
so all sum to 1 for 'E'
'''
import numpy as np
from functools import reduce
from collections import OrderedDict
import matplotlib.pyplot as plt
import networkx as nx

try:
    # import tokentranslator.translator.grammar.cyk as cyk
    import tokentranslator.translator.grammar.backward as backward
except:
    print("Import tokentranslator fail, trying using local folder")
    import backward as backward

grammar1 = [('E', 'AE'), ('E', 'BE'),
            ('E', 'e'), ('A', 'a'), ('B', 'b'),
            ('B', 'c')]


# not used:
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


def em(sents=[list('aaabbe')], grammar=grammar1,
       ts=None, ts_orign=None, debug=False):

    '''
    - ``ts`` -- init model, initiated with probabilities
    if not given, will be generated equaly for all rules
    (to sum to 1).
    '''
    # init model:
    if ts is None:
        ts = backward.get_probabilities_all(grammar)

    if debug:
        print("ts:")
        for rule in ts:
            print(rule, ": ", ts[rule])

    gc = {}
    gc_unterm = {}
    gc_term = {}
    
    # backup in case of next result c will fail:
    ts_old = dict((rule, ts[rule]) for rule in ts)
    # ts_orign = dict((rule, ts[rule]) for rule in ts)

    xs = np.zeros((len(ts), len(sents)))
    for n, sent in enumerate(sents):
        
        ires = inside(sent=sent, grammar=grammar, ts=ts,
                      debug=False)
        alpha = ires[1]
        if debug:
            print("\ninside(sent=%s):" % "".join(sent))
            print(ires[0])
            print("outside(sent=%s):" % "".join(sent))

        ores = outside(alpha, sent=sent, grammar=grammar, ts=ts,
                       debug=False)
        if debug:
            print("p: ", ores[0])
            print("em_step(sent=%s):" % "".join(sent))
        
        beta = ores[1]

        _, _, c, c_unterm, c_term = em_step(alpha, beta, sent=sent,
                                            grammar=grammar, ts=ts, debug=False)
        # FOR update c:
        if debug:
            print("c:")
            for rule in c:
                print(rule, ": ", c[rule])

            print("step %d" % n)
            print("sent: ", sent)
            print("c['E']: ", c['E'], type(c['E']), np.nan_to_num(c['E']))
            print(np.nan in c.values())
        if np.nan in c.values() or np.nan_to_num(c['E']) <= 0:
            # print("ts:")
            # print(ts)
            ts = ts_old
            continue

        for idx in c:
            if idx not in gc:
                gc[idx] = c[idx]
            else:
                gc[idx] += c[idx]

        for idx in c_unterm:
            if idx not in gc_unterm:
                gc_unterm[idx] = c_unterm[idx]
            else:
                gc_unterm[idx] += c_unterm[idx]
                
        for idx in c_term:
            if idx not in gc_term:
                gc_term[idx] = c_term[idx]
            else:
                gc_term[idx] += c_term[idx]
        # END FOR

        # FOR update ts:
        ts = {}
        for (v, yz) in gc_unterm:
            if gc[v] != 0:
                ts[(v, yz)] = gc_unterm[(v, yz)]/float(gc[v])
            else:
                ts[(v, yz)] = 0

        for (v, a) in gc_term:
            if gc[v] != 0:
                ts[(v, a)] = gc_term[(v, a)]/float(gc[v])
            else:
                ts[(v, a)] = 0
        
        for idx, rule in enumerate(ts):
            xs[idx, n] = np.abs(ts[rule]-ts_orign[rule])
        
        ts_old = dict((rule, ts[rule]) for rule in ts)
        # END FOR

        if debug:
            print("step %d" % n)
            print("c:")
            for rule in c:
                print(rule, ": ", c[rule])
            print("c_unterm:")
            for rule in c_unterm:
                print(rule, ": ", c_unterm[rule])
            print("c_term:")
            for rule in c_term:
                print(rule, ": ", c_term[rule])
            
            print("gc:")
            for rule in gc:
                print(rule, ": ", gc[rule])
            
            print("gc_unterm:")
            for rule in gc_unterm:
                print(rule, ": ", gc_unterm[rule])
            
            print("gc_term:")
            for rule in gc_term:
                print(rule, ": ", gc_term[rule])

            # print("ts:")
            # for rule in ts:
            #     print(rule, ": ", ts[rule])
        
    return((ts, xs))


def em_step(alpha, beta, sent=list('aaabbe'), grammar=grammar1,
            ts=None, debug=False):

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

    if ts is None:
        ts = backward.get_probabilities_all(grammar)

    if debug:
        print("names:")
        print([nodes_names[v] for v in range(M)])

        print("ts:")
        print(ts)

        print("alpha:")
        print(alpha)

        print("beta:")
        print(beta)

    p_sent_gm = alpha[nodes_names.index('E'), 0,  L-1]
    c = np.array([(alpha[v][triu] * beta[v][triu]).sum()
                  for v in range(M)])
    c1 = c
    # import pdb; pdb.set_trace()
    if debug:
        print("\ncounts c:")
        print(c)
    c = c/p_sent_gm
    if debug:
        print("p_sent_gm:")
        print(p_sent_gm)
        print("c:")
    c = dict(zip(nodes_names, c))
    if debug:
        print(c)
    
    c_unterm = {}
    
    for (v, yz) in ts:
        if len(yz) > 1:
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
    for (v, a) in ts:
        if len(a) == 1:
            v_idx = nodes_names.index(v)
            # print((v, a, v_idx))
            # print([beta[v_idx, i, i]
            #        for i, x in enumerate(sent)
            #        if x == a])
            c_term[(v, a)] = sum([beta[v_idx, i, i] * ts[v, x]
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
            t[(v, yz)] = 0

    e = {}
    for (v, a) in c_term:
        if c[v] != 0:
            e[(v, a)] = c_term[(v, a)]/float(c[v])
        else:
            e[(v, a)] = 0
    return((t, e, c,  c_unterm, c_term))


def outside(alpha, sent=list('aaabbe'), grammar=grammar1,
            ts=None, debug=False):

    '''
    Calculate outside values beta i.e:
    beta[i, j, v] is probability of parse tree, rooted at v
    for seq x excluding all parse subtrees for seq x_{i},...x_{j}.
    p(sent|grammar) = \sum_{v \in [1 .. M]} beta(i, i, v) * e_{v}(x_{i})
    
    (see. Durbin p. 255)
    rules in grammar must be sorted in decreasing order'''

    L = len(sent)

    if ts is None:
        ts = backward.get_probabilities_all(grammar)
        
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
                    if len(rule[1]) > 1:
                        # if unterminal rule exist:
                        if (v == rule[1][1] and i > 0):
                            
                            y = rule[0]
                            z = rule[1][0]
                            y_idx = nodes_names.index(y)
                            z_idx = nodes_names.index(z)
                            beta[v_idx, i, j] += ((alpha[z_idx, 0:i, i-1]
                                                   * beta[y_idx, 0:i, j])
                                                  .sum() * ts[rule])
                        if (v == rule[1][0] and j < L-1):
                            
                            y = rule[0]
                            z = rule[1][1]
                            y_idx = nodes_names.index(y)
                            z_idx = nodes_names.index(z)
                            beta[v_idx, i, j] += ((alpha[z_idx, j+1, j+1:L]
                                                   * beta[y_idx, i, j+1:L])
                                                  .sum() * ts[rule])
    if debug:
        p = []
        for i, x in enumerate(sent):
            pp_tmp = [(v_idx, i, v, x)
                      for v_idx, v in enumerate(nodes_names)
                      if (v, x) in ts]

            p_tmp = [beta[v_idx, i, i] * ts[v, x]
                     for v_idx, v in enumerate(nodes_names)
                     if (v, x) in ts]
            print("p_tmp, x=%s" % x)
            print(pp_tmp)
            print(p_tmp)

            print("beta[:, %d, %d]:" % (i, i))
            print(beta[:, i, i])

            p.append(sum(p_tmp))
            # p.extend(p_tmp)
    else:
    
        p = [sum([beta[v_idx, i, i] * ts[v, x]
                  for v_idx, v in enumerate(nodes_names)
                  if (v, x) in ts])
             for i, x in enumerate(sent)]

    return((p, beta))
    

def cyk(sent, i, j, v, gammas, tildas, grammar=None, ts=None):
    '''p.257 cyk parse algorithm for SCFG'''
    gamma = cyk
    fgrammar = factorize(grammar)

    if ts is None:
        ts = backward.get_probabilities_all(grammar)
        print("probs:")
        for rule in ts:
            print(rule, ": ", ts[rule])

    if (i, j, v) in gammas:
        return(gammas[(i, j, v)])

    # print("v: ", v)
    # print("i,j:", (i, j))
    # print("sent[i: j]: ", sent[i: j])

    if i == j:
        # print(ts)
        # print(v)
        # print(sent[i])
        # print(ts[(v, sent[i])] if (v, sent[i]) in ts else 0)
        gamma_new = ts[(v, sent[i])] if (v, sent[i]) in ts else 0
        tilda_new = (0, 0, 0)
        gammas[(i, j, v)] = gamma_new
        tildas[(i, j, v)] = tilda_new
        return(gamma_new)
    
    # print("tilda")
    # print(tildas)
    # print("gammas")
    # print(gammas)
    
    iterations = [
        (
            gamma(sent, i, k, yz[0], gammas, tildas,
                  grammar=grammar, ts=ts)
            * gamma(sent, k+1, j, yz[1], gammas, tildas,
                    grammar=grammar, ts=ts)
            * ts[(v, yz)], (yz[0], yz[1], k))
        for k in range(i, j) for yz in fgrammar[v]
        if len(yz) > 1]
    if len(iterations) == 0:
        # gammas[(i, j, v)] = 0
        # tildas[(i, j, v)] = 
        return(0)
    gamma_new, tilda_new = max(iterations, key=lambda x: x[0])
    
    gammas[(i, j, v)] = gamma_new
    tildas[(i, j, v)] = tilda_new
    return(gamma_new)

    
def make_parse_tree(sent, tildas, gammas, idx=None, g=None, level=""):
    if idx is None:
        L = len(sent)-1
        idx = (0, L, "E")
        g = nx.Graph()

    print("idx:", idx)    
    y, z, k = tildas[idx]
    i, j, v = idx
    
    if y == z == k == 0:
        # i == j in this case:
        # v_new = v+" "+"%.3f" % gammas[(i, j, v)]
        v_new = v+level
        g.add_edge(v_new, str(sent[i])+" "+level+" %.3f" % gammas[(i, j, v)])
        return(g)

    v_new = v+level+"->"+y+" "+z+" "+"%.3f" % gammas[(i, k, y)]
    g.add_edge(v+level, v_new)
    g.add_edge(v+level, y+level+"l")
    # , weight=gammas[(i, k, y)]
    # v_new = v+"->"+y+" "+z+" "+"%.3f" % gammas[(k+1, j, z)]
    # g.add_edge(v+level, v_new)
    g.add_edge(v+level, z+level+"r")
    g = make_parse_tree(sent, tildas, gammas,
                        idx=(i, k, y), g=g, level=level+"l")
    g = make_parse_tree(sent, tildas, gammas,
                        idx=(k+1, j, z), g=g, level=level+"r")
    return(g)


def inside_rec(sent, i, j, v, grammar=None, ts=None):
    '''Same as inside, but recursive'''

    alpha = inside_rec
    
    fgrammar = factorize(grammar)

    if ts is None:
        ts = backward.get_probabilities_all(grammar)

    if i == j:
        return(ts[(v, sent[i])] if (v, sent[i]) in ts else 0)

    # range(i, j) = [i, ..., j-1]:
    res = sum([sum([alpha(sent, i, k, yz[0],
                          grammar=grammar, ts=ts)
                    * alpha(sent, k+1, j, yz[1],
                            grammar=grammar, ts=ts)
                    for k in range(i, j)])*ts[(v, yz)]
               for yz in fgrammar[v] if len(yz) > 1])
    return(res)


def outside_rec(sent, i, j, v, grammar=None, ts=None):
    '''i, j from 0 to L-1'''
    alpha = lambda i, j, v: inside_rec(sent, i, j, v,
                                       grammar=grammar, ts=ts)
    beta = lambda i, j, v: outside_rec(sent, i, j, v,
                                       grammar=grammar, ts=ts)

    L = len(sent)-1
    
    if ts is None:
        ts = backward.get_probabilities_all(grammar)

    fgrammar = factorize(grammar)
    
    if i == 0 and j == L:
        return(1 if v == 'E' else 0)

    bl = sum([sum([alpha(k, i-1, zv[0])*beta(k, j, y)
                   for k in range(0, i)])
              * ts[(y, zv)] for y in fgrammar
              for zv in fgrammar[y] if len(zv) > 1 and zv[1] == v])

    br = sum([sum([alpha(j+1, k, vz[1])*beta(i, k, y)
                   for k in range(j+1, L+1)])
              * ts[(y, vz)] for y in fgrammar
              for vz in fgrammar[y] if len(vz) > 1 and vz[0] == v])
    return(bl+br)


def inside(sent=list('aaabbe'), grammar=grammar1,
           ts=None, debug=False):
    '''
    Calculate inside values alpha i.e:
    alpha[i, j, v] is probability of parse tree, rooted at v
    for seq x.
    p(sent|grammar) = alpha(1, L, 1)
    
    (see. Durbin p. 254)
    (Lari-Young p. 3)
    '''

    L = len(sent)

    if ts is None:
        ts = backward.get_probabilities_all(grammar)

    fgrammar = factorize(grammar)
    if debug:
        print("fgrammar:")
        print(fgrammar)
        print("grammar:")
        print(grammar)
    # ufgrammar = factorize(cyk.get_rules(grammar1, term=False))
    
    M = len(fgrammar)

    # init:
    alpha = np.zeros((M, L, L))
    for i in range(L):
        for v_idx, v in enumerate(fgrammar):

            # if there is emission for terminal sent[i]
            # from node v:
            if (v, sent[i]) in ts:
                alpha[v_idx, i, i] = ts[(v, sent[i])]

    
    nodes_names = list(fgrammar.keys())
    # indexes of term/unters is same as in nodes_names:
    rules_idxs = [(v_idx, v) for v_idx, v in enumerate(fgrammar)]
    rules_idxs.reverse()
    if debug:
        print("\nfgrammar:")
        for rule in fgrammar:
            print(rule + ": " + str(fgrammar[rule]))
        print("\nts:")
        for rule in ts:
            print(rule, ": ", ts[rule])
        print("nodes_names:", nodes_names)
        print("v_idx, v:", rules_idxs)
        print("alpha[%s, i, i]:" % str(nodes_names))
        for i in range(L):
            M = len(nodes_names)
            for line in zip(["i = "]*M, [i]*M, ["v = "]*M, nodes_names,
                            ["alpha[v, i, i] = "]*M, alpha[:, i, i]):
                print(line)
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
                        
                        '''
                        if debug:
                            #  and (v, yz) == ('E', 'BE')
                            print("fgrammar[v]:", fgrammar[v])
                            print("\nsent:", "".join(sent))
                            print("%s->%s" % (v, yz))
                            print("nodes_names:", nodes_names)
                            print("v_idx, v:", rules_idxs)
                            print("alpha[v, i, j]")
                            print("alpha[%d, %d, %d]:" % (v_idx, i, j),
                                  alpha[v_idx, i, j])
                            print("= y ======")
                            print("alpha[y, i, i:j-1+1]")
                            print("alpha[%d, %d, %d:%d]: " % (y_idx, i, i, j-1+1),
                                  alpha[y_idx, i, i:j-1+1])
                            # print("alpha[y, i, :]")
                            # print("alpha[%d, %d, :]" % (y_idx, i),
                            #       alpha[y_idx, i, :])
                            print("= z ======")
                            print("alpha[z, i+1:j+1, j]")
                            print("alpha[%d, %d:%d, %d]: " % (z_idx, i+1, j+1, j),
                                  alpha[z_idx, i+1:j+1, j])
                            print("ts[(%s, %s)]=" % (v, yz), ts[(v, yz)])
                            # print("alpha[z, :, j]")
                            # print("alpha[%d, :, %d]" % (z_idx, j),
                            #       alpha[z_idx, :, j])
                        '''
            if debug:
                print()
                print("sent:", "".join(sent))
                print("nodes_names:", nodes_names)
                print("rules_idxs:", rules_idxs)
                print("alpha[%s, i, j] = " % str(nodes_names),
                      "alpha[:, %d, %d] = " % (i, j),
                      "alpha[:, '%s'] =" % ("".join(sent[i:j+1])))
                print(alpha[:, i, j])
    # return p(sent|grammar) = alpha['E', 0, len(sent)]=
    #           = alpha['E', 0, len(sent)-1] (due to range):
    if debug:
        print("\nalpha(E, 0, %d)" % (L-1), " = ",
              alpha[nodes_names.index('E'), 0,  L-1])
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
    p_sent, alpha = inside(sent=list('be'), grammar=backward.grammar2,
                           debug=False)
    print("p(sent='be') = ", p_sent)
    print("\nalpha")
    print(alpha)
    
    print("inside(sent=list('ae')):")
    p_sent, alpha = inside(sent=list('ae'), grammar=backward.grammar3, 
                           debug=False)
    print("p(sent='ae') = ", p_sent)
    print("\nalpha")
    print(alpha)
    
    print("inside(sent=list('aae')):")
    p_sent, alpha = inside(sent=list('aae'), grammar=backward.grammar3,
                           debug=True)
    print("p(sent='aae') = ", p_sent)
    print("\nalpha")
    print(alpha)


def test_outside():
    sent = list('aae')
    grammar = backward.grammar3
    print("inside(sent=list('aae')):")
    p_sent, alpha = inside(sent=sent, grammar=grammar,
                           debug=False)
    print("p(sent='aae') = ", p_sent)
    print("\nalpha")
    print(alpha)

    p, beta = outside(alpha, sent=sent, grammar=grammar,
                      debug=True)
    print("p = ", p)
    print("beta:")
    print(beta)


def test_outside_rec():
    print()
    res = outside_rec(list('aae'), 0, 0, 'E', grammar=backward.grammar3)
    print("res outside('aae', 0, 0, 'E'):", res)
    res = outside_rec(list('aae'), 0, 0, 'A', grammar=backward.grammar3)
    print("res outside('aae', 0, 0, 'A'):", res)
    res = outside_rec(list('aae'), 0, 0, 'B', grammar=backward.grammar3)
    print("res outside('aae', 0, 0, 'B'):", res)

    print()
    res = outside_rec(list('aae'), 1, 2, 'E', grammar=backward.grammar3)
    print("res outside('aae', 1, 2, 'E'):", res)
    res = outside_rec(list('aae'), 1, 2, 'A', grammar=backward.grammar3)
    print("res outside('aae', 1, 2, 'A'):", res)
    res = outside_rec(list('aae'), 1, 2, 'B', grammar=backward.grammar3)
    print("res outside('aae', 1, 2, 'B'):", res)

    print()
    res = inside_rec(list('aae'), 0, 2, 'E', grammar=backward.grammar3)
    print("res inside('aae', 0, 2, 'E'):", res)


def test_em(N=3, sent_len=5, max_sent_len=100, plot=True):
    
    '''
    ts3 = {
        ('E', 'AE'): 1/4, ('E', 'BE'): 1/4, ('E', 'e'): 3/8,
        ('E', 'a'): 1/8,
        ('A', 'AE'): 2/3, ('A', 'a'): 8/27, ('A', 'e'): 1/27,
        ('B', 'BE'): 2/3, ('B', 'AE'): 1/81, ('B', 'b'): 26/81
    }
    '''
    ts3 = {
        ('E', 'AE'): 1/4, ('E', 'BE'): 1/4, ('E', 'e'): 3/8,
        ('E', 'a'): 1/8,
        ('A', 'AE'): 8/27, ('A', 'a'): 2/3, ('A', 'e'): 1/27,
        ('B', 'BE'): 8/27, ('B', 'AE'): 1/27, ('B', 'b'): 2/3
    }

    ts31 = {
        ('E', 'AE'): 1/4, ('E', 'BE'): 1/4, ('E', 'e'): 1/2,
        
        ('A', 'AE'): 8/27, ('A', 'a'): 2/3, ('A', 'e'): 1/27,
        ('B', 'BE'): 26/81, ('B', 'AE'): 1/81, ('B', 'b'): 2/3
    }
    grammar31 = [('E', 'AE'), ('E', 'BE'), ('E', 'e'),
                 ('A', 'AE'), ('A', 'a'), ('A', 'e'),
                 ('B', 'BE'), ('B', 'AE'), ('B', 'b')]

    '''
    grammar4 = [('E', 'AE'),  ('E', 'e'), ('E', 'a'),
                ('A', 'AE'), ('A', 'a'), ('A', 'e')]
    ts3 = {
        ('E', 'AE'): 1/2,  ('E', 'e'): 3/8, ('E', 'a'): 1/8,
        ('A', 'AE'): 2/3, ('A', 'a'): 8/27, ('A', 'e'): 1/27
    }
    grammar = grammar4
    '''
    grammar = grammar31
    ts_orign = ts31
    # grammar = backward.grammar3
    lsents = []
    idx_max_len = 0
    max_len = 0
    print("=======================")
    while len(lsents) < N:
        lsent = backward.gen_sent(grammar, sent_len, probs=ts_orign)

        # exclude triviality:
        len_lsent = len(lsent)
        if (len_lsent > 1 and 'a' in lsent and 'b' in lsent
            and len_lsent < max_sent_len):
            lsents.append(lsent)
            
            # max len sent in front:
            if len_lsent > max_len:
                idx_max_len = len(lsents)-1
                max_len = len_lsent

    # max len sent in front:
    # for all rules initiated at first
    # with no zero probabilities:
    max_sent = lsents.pop(idx_max_len)
    lsents.insert(0, max_sent)

    ts, xs = em(sents=lsents, grammar=grammar,
                ts_orign=ts_orign, debug=True)
    if plot:
        plt.plot(xs.T)
        plt.show()

    print("\noriginal ts:")
    for rule in ts_orign:
        print(rule, ": ", ts_orign[rule])
    print("lerned ts:")
    for rule in ts_orign:
        if rule in ts:
            print(rule, ": ", ts[rule])
        else:
            pass
            # print(rule, ": ", es[rule])
    for lsent in lsents[:7]:
        print("".join(lsent))
    '''
    grammar2 = [('E', 'AE'), ('E', 'BE'),
                ('A', 'BA'), ('A', 'BE'),
                ('E', 'e'), ('A', 'a'), ('B', 'b'),
                ('B', 'c')]

    ts2 = {('E', 'AE'): 0.5, ('E', 'BE'): 0.5,
           ('A', 'BA'): 0.5, ('A', 'BE'): 0.5,
           ('E', 'e'): 1.0, ('A', 'a'): 1.0,
           ('B', 'b'): 0.5, ('B', 'c'): 0.5}

    grammar3 = [('E', 'AE'), ('E', 'BE'),
                ('A', 'BA'), ('A', 'BE'),
                ('B', 'AB'), ('B', 'AE'),

                ('E', 'e'), ('E', 'a'), ('E', 'b'), ('E', 'c'),
                ('A', 'e'), ('A', 'a'), ('A', 'b'), ('A', 'c'),
                ('B', 'e'), ('B', 'a'), ('B', 'b'), ('B', 'c')]

    ts3 = {('E', 'AE'): 0.5, ('E', 'BE'): 0.5,
           ('A', 'BA'): 0.5, ('A', 'BE'): 0.5,
           ('B', 'AB'): 0.5, ('B', 'AE'): 0.5,
           ('E', 'e'): 0.91, ('E', 'a'): 0.03,
           ('E', 'b'): 0.03, ('E', 'c'): 0.03,

           ('A', 'a'): 0.7, ('A', 'b'): 0.2,
           ('A', 'c'): 0.05, ('A', 'e'): 0.05,

           ('B', 'a'): 0.2, ('B', 'b'): 0.7,
           ('B', 'c'): 0.05, ('B', 'e'): 0.05}
    '''


def test_em_step(sent=list('aaabbe'), grammar=grammar1,
                 ts=None):

    ires = inside(sent=sent, grammar=grammar, ts=ts,
                  debug=False)
    alpha = ires[1]
    print("inside(sent=%s):" % "".join(sent))
    print(ires[0])
    print("outside(sent=%s):" % "".join(sent))
    ores = outside(alpha, sent=sent, grammar=grammar,
                   ts=ts, debug=True)
    print(ores[0])
    beta = ores[1]

    print("em_step(sent=%s):" % "".join(sent))
    t, e, c, c_unterm, c_term = em_step(alpha, beta, sent=sent,
                                        grammar=grammar, ts=ts,
                                        debug=True)
    # test()
    # print("\nt:")
    # print(t)
    # print("\ne:")
    # print(e)
    print("c")
    for rule in c:
        print(rule, ": ", c[rule])
    print("c_unterm")
    for rule in c_unterm:
        print(rule, ": ", c_unterm[rule])

    print("c_term")
    for rule in c_term:
        print(rule, ": ", c_term[rule])
    

def test_em_step_rec(sent=list('aae'), grammar=backward.grammar3):
    L = len(sent)
    fgrammar = factorize(grammar)
    # print("fgrammar")
    # print(fgrammar)
    ts = backward.get_probabilities_all(grammar)
    # print("ts:")
    # print(ts)

    alpha = lambda i, j, v: inside_rec(sent, i, j, v, grammar=grammar)
    beta = lambda i, j, v: outside_rec(sent, i, j, v, grammar=grammar)

    p = alpha(0, L-1, 'E')
    print("p = ", p)

    c = OrderedDict()
    ps = OrderedDict()
    for v in fgrammar:
        # this include term and unterm rules:
        c[v] = 1/p * sum([alpha(i, j, v) * beta(i, j, v)
                          for i in range(L) for j in range(L)])

        for yz in fgrammar[v]:
            if len(yz) == 2:
                
                c[(v, yz)] = 1/p * sum([
                    beta(i, j, v) * alpha(i, k, yz[0])
                    * alpha(k+1, j, yz[1]) * ts[(v, yz)]
                    for i in range(L-1)
                    for j in range(i+1, L)
                    for k in range(i, j)])
            if len(yz) == 1:
                # print(yz)
                # print("alpha(i, i, %s)" % yz,
                #       [alpha(i, i, v) for i in range(L) if sent[i] == yz])
                # print("beta(i, i, %s)" % yz,
                #       [beta(i, i, v) for i in range(L) if sent[i] == yz])
                c[(v, yz)] = 1/p * sum([
                    alpha(i, i, v)*beta(i, i, v)
                    for i in range(L) if sent[i] == yz])
            ps[(v, yz)] = c[(v, yz)]/c[v] if c[v] > 0 else 'inf' 
    print("c:")
    for rule in c:
        print(rule, ": ", c[rule])
    # print("ps:")
    # print(ps)


def test_cyk():
    gammas = {}
    tildas = {}
    res = cyk(list('aae'), 0, 2, 'E', gammas, tildas, grammar=backward.grammar3)
    print(res)
    for idx in gammas:
        print(idx, ": ", gammas[idx], "; ", tildas[idx])
    g = make_parse_tree(list('aae'), tildas, gammas)
    print("g:")
    print(g.edges)
    layout = nx.spring_layout(g)
    nx.draw(g, pos=layout)
    nx.draw_networkx_labels(g, layout, font_size=12)
    plt.show()


grammar_subjunctive = [

    ("E", ("CONDIF",)), ("E", ("CONDUNLESS",)),
    ('CONDIF', ('IF', 'A', 'B')), 
    ('CONDUNLESS', ('UNLESS', 'B', 'A', )),
    ('CONDUNLESS', ('B', 'UNLESS','A')),
    ('A', ('OTHERS', 'PAST', 'OTHERS')), ('A', ('OTHERS', 'HPAST', 'OTHERS')),
    ('B', ('OTHERS', 'WOULD', 'OTHERS')), ('B', ('OTHERS', 'WOULD','HPAST','OTHERS')),

    ('IF', ('if',)), ('IF', ('but_for',)), ('IF', ('even_if',)),
    ('IF', ('even though', )),
    ('IF', ('if where not for', )),

    ('UNLESS', ('unless',)), ('UNLESS', ('in_case',)),
    ('UNLESS', ('given',)), ('UNLESS', ('given_that',)),
    ('UNLESS', ('provided',)), ('UNLESS', ('provided_that',)),
    ('UNLESS', ('providing',)), ('UNLESS', ('providing_that',)),
    ('UNLESS', ('granted',)), ('UNLESS', ('granted_that',)),

    ('HPAST', ('HAVE', 'PAST')), ('HPAST', ('HAVE', 'PAST')),
    ('HAVE', ('have',)), ('HAVE', ('had',)),

    ('PAST', ('past', )),   
    # ('PAST', ('OTHERS', )),  # because we are lazy

    ('WOULD', ('would',)), ('WOULD', ('should',)),
    ('WOULD', ('could',)), ('WOULD', ('might',)),

    ('OTHERS', ('e',))]


def test_cyk2():
    from cyk import cyk as original_cyk
    import grammars as gm0
    grammar_subjunctive_cnf = gm0.gm_to_cnf(grammar_subjunctive)
    sent = ['e', 'should', 'e', 'unless', 'e', 'past', 'e']
    # sent = [('e',), ('should',), ('e',), ('unless',), ('e',)]
    res = original_cyk(goal=sent, grammar=grammar_subjunctive_cnf)
    print("original_cyk:")
    print(res)


def test_cyk1():

    import grammars as gm0
    
    grammar_subjunctive_cnf = gm0.gm_to_cnf(grammar_subjunctive)
    # grammar_subjunctive_cnf

    gammas = {}
    tildas = {}
    
    sent = [('e',), ('should',), ('e',), ('unless',), ('e',), ('past',), ('e',)]
    # sent = [('e',), ('should',), ('e',), ('unless',), ('e',)]
    L = len(sent) - 1
    res = cyk(sent, 0, L, 'E', gammas, tildas,
              grammar=grammar_subjunctive_cnf,)
    print("P(sent|grammar31)", res)
    print("\ngammas, tildas:")
    for idx in gammas:
        print(idx, ": ", gammas[idx], "; ", tildas[idx])
    g = make_parse_tree(sent, tildas, gammas)
    print("\ngraph.edges:")
    print(g.edges)
    layout = nx.spring_layout(g)
    nx.draw(g, pos=layout)
    nx.draw_networkx_labels(g, layout, font_size=12)
    plt.show()


if __name__ == '__main__':
    '''
    ('E', 'AE') :  0.25
    ('E', 'BE') :  0.25
    ('E', 'e') :  0.375
    ('E', 'a') :  0.125
    ('A', 'AE') :  0.6666666666666666
    ('A', 'a') :  0.2962962962962963
    ('A', 'e') :  0.037037037037037035
    ('B', 'BE') :  0.6666666666666666
    ('B', 'AE') :  0.037037037037037035
    ('B', 'b') :  0.2962962962962963
    lerned ts:
    ('E', 'AE') :  0.2604149864053485
    ('E', 'BE') :  0.25925396909520976
    ('E', 'e') :  0.3515752789393456
    ('E', 'a') :  0.12875576556009619
    ('A', 'AE') :  0.3309651411456314
    ('A', 'a') :  0.39000000035894744
    ('A', 'e') :  0.2790348584954211
    ('B', 'BE') :  0.3146000004726746
    ('B', 'AE') :  0.37645226065140386
    ('B', 'b') :  0.3089477388759218

    original ts:
    ('E', 'AE') :  0.25
    ('E', 'BE') :  0.25
    ('E', 'e') :  0.5
    ('A', 'AE') :  0.2962962962962963
    ('A', 'a') :  0.6666666666666666
    ('A', 'e') :  0.037037037037037035
    ('B', 'BE') :  0.32098765432098764
    ('B', 'AE') :  0.012345679012345678
    ('B', 'b') :  0.6666666666666666
    lerned ts:
    ('E', 'AE') :  0.40395400862023734
    ('E', 'BE') :  0.29560533619060136
    ('E', 'e') :  0.30044065518916113
    ('A', 'AE') :  0.13436336117688144
    ('A', 'a') :  0.6212361052628395
    ('A', 'e') :  0.2444005335602786
    ('B', 'BE') :  0.28461777923011555
    ('B', 'AE') :  0.06136185609043002
    ('B', 'b') :  0.6540203646794545
    '''
    # test_cyk2()
    test_cyk1()
    # test_cyk()
    # test_em(N=20, sent_len=7)
    # sent = list('eabaeeabbbeebeeeaaebbbbeeae')
    # sent = list('aae')
    # test_em_step(sent=sent, grammar=backward.grammar3)
    # print("\nrec:")
    # test_em_step_rec(sent=sent, grammar=backward.grammar3)
    
    # test_em_step(sent=list('aae'), grammar=backward.grammar3)
    # test_em_step_rec(sent=list('aae'), grammar=backward.grammar3)

    # test_outside()
    # test_outside_rec()

    # test_inside()
    '''

    '''
    '''
    ires = inside(sent=list('aabcbe'), grammar=grammar3,
                  ts=ts3, es=es3,
                  debug=True)
    print("inside:")
    print(ires[1])
    '''
    # test_em_step(sent=list('aabcbe'), grammar=grammar3, ts=ts3, es=es3)
    # test_em_step(sent=list('aabcbe'), grammar=grammar1)
    # test_em_step(sent=list('(a+a)*a'), grammar=cyk.grammar)

