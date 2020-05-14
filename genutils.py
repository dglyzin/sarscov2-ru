#utilities
def readgenfile(filename):
    count = 0 
    with open(filename,'r') as f:
        header = ''
        gotheader = False
        genomes = []
        for line in f:
            #print(line[0])
            if gotheader:
                genomes.append({'header':header.rstrip(), 'sequence':line.rstrip()})
                gotheader = False
            elif line[0]=='>':
                header = line[1:]
                gotheader = True        
            count += 1
    print ("lines read: ", count)
    print ("sequences read: ", len(genomes))
    return genomes

def savegenfile(filename, genomes):
    count = 0
    with open(filename,'w') as f:
        for genome in genomes:
            f.write('>'+genome["header"])
            f.write('\n')
            f.write(genome["sequence"])
            f.write('\n')
            count += 1
    print('sequences saved:', count)

def filtergenomes(genomes, filter):
    return [genome for genome in genomes if filter in genome["header"] ]

def filtergenomesbyfile(genomes, filterfilename):
    '''
    returns filtered list of genomes with headers containing lines from filterfilename
    '''
    filteredgenomes = []
    with open(filterfilename, "r") as f:
        for line in f:
            line = line.rstrip()
            filtered = filtergenomes(genomes,line)
            for genome in filtered:
                filteredgenomes.append(genome)
            if len(filtered)==0:    
                print("Not found! ", line)            
    return filteredgenomes

def filtergensize(genomes, direction, size):
    if direction=="greater": 
        return [genome for genome in genomes if len(genome["sequence"])>size ]
    elif direction=="smaller":
        return [genome for genome in genomes if len(genome["sequence"])<size ]
    else:
        return []
    
def getgensampleparams(genomes):
    '''
    returns mean and variance of sequence samples length
    '''
    glen = len(genomes)
    if glen>0:
        total = 0
        for genome in genomes: 
            total += len(genome["sequence"])
        mean = total/glen
        total = 0
        for genome in genomes: 
            diff = len(genome["sequence"])-mean
            #print(diff*diff)
            total += diff*diff
        variance = total/glen
    else:
        mean = variance = 0.0
    return mean, variance