# The script generating random peptides from a predefined distribution for the Monte Carlo tests. The output is a text file with peptides.

import sys, os, itertools, re, argparse
from aa_constants import AA_FREQ, AA_GROUPS
from ml_classes import RandomPeptideGenerator
from thesis_constants import MC_patterns
from functions import error

#Processing command line arguments
usage = "Usage: %(prog)s [options] -p POSITIVE_SET_FILE -n NEGATIVE_SET_FILE"
parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('-op', dest = 'positive_set_file', required = True, help = 'Output text file with positive mock dataset')
parser.add_argument('-on', dest = 'negative_set_file', required = True, help = 'Output text file with negative mock dataset')
parser.add_argument('-np', dest = 'count_positive', type = int, required = True, help = 'Number of peptides in the positive set')
parser.add_argument('-nn', dest = 'count_negative', type = int, required = True, help = 'Number of peptides in the negative set')
parser.add_argument('-l', dest = 'length', type = int, default = 20, help = 'Peptide length')
parser.add_argument('-s', dest = 'series', required = True, help = 'Monte Carlo test series [for the thesis]')
if __name__ == "__main__":
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    opt = parser.parse_args()
    opt.positive_set_file = os.path.abspath(opt.positive_set_file)
    opt.negative_set_file = os.path.abspath(opt.negative_set_file)
    if (opt.count_positive < 10) or (opt.count_negative < 10):
        error('There must be at least 10 peptides in each dataset')
    if (len(opt.series) != 1) or (opt.series not in MC_patterns.keys()):
        error('Invalid Monte Carlo series')

def generate_all_dimers(linker_len_max = 3):
    """Funciton to generate all possible dimers to measure"""
    dimer_regexes = []
    for group0_aas, group1_aas, linker_len in itertools.product(AA_GROUPS.values(), AA_GROUPS.values(), range(linker_len_max)):
        dimer_regex = '[{}]{}[{}]'.format(group0_aas, '.' * linker_len, group1_aas)
        dimer_regexes.append(re.compile(dimer_regex))
    return dimer_regexes

#dimer_regexes = generate_all_dimers()
enriched_dimers = MC_patterns[opt.series]
peptide_generator = RandomPeptideGenerator(AA_FREQ, length = opt.length, enriched_dimers = enriched_dimers)
positive_set = peptide_generator.generate_random_peptides(n = opt.count_positive, with_dimers = True)
negative_set = peptide_generator.generate_random_peptides(n = opt.count_negative, with_dimers = False)
with open(opt.positive_set_file, 'w') as ofile:
    ofile.write('\n'.join(positive_set))
with open(opt.negative_set_file, 'w') as ofile:
    ofile.write('\n'.join(negative_set))