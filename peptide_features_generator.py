# The scriot converting the input text (FASTA) file with peptides to a NumPy array of features.

import os, sys, argparse, re, itertools
import numpy as np
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aa_constants import AA_GROUPS
from functions import error

#Global variables
g_aas = 'ACDEFGHIKLMNPQRSTVWY'
g_features = OrderedDict()
for aa in g_aas: #single amino acid occurrence features
    g_features[aa] = re.compile(aa)
for group, aas in AA_GROUPS.items(): #amino acid group occurrence features
    g_features[group] = re.compile('[{}]'.format(aas))
for group0, group1, linker_len in itertools.product(AA_GROUPS.keys(), AA_GROUPS.keys(), range(4)): #fuzzy pattern occurrence features
    aas0 = AA_GROUPS[group0]
    aas1 = AA_GROUPS[group1]
    feature_name = '{}-{}{}'.format(group0, 'x-' * linker_len, group1)
    g_features[feature_name] = re.compile('[{}]{}[{}]'.format(aas0, '.' * linker_len, aas1))
g_features['repeat_stretch'] = re.compile(r'((\w)\2+)') #a low complexity feature
g_features['repeat_pair'] = re.compile(r'(?=((\w)\2))') #a low complexity feature

#Processing command line arguments
usage = "Usage: %(prog)s [options] -i INPUT_FILE -o OUTPUT_FILE"
parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('-i', dest = 'input_file', required = True, help = 'Input file with peptides')
parser.add_argument('-o', dest = 'output_file', required = True, help = 'Output file with Numpy array of features')
parser.add_argument('-names', dest = 'output_names', default = '', help = 'Output file with Numpy array of feature names')  
if __name__ == "__main__":
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    opt = parser.parse_args()
    opt.input_file = os.path.abspath(opt.input_file)
    if not os.path.isfile(opt.input_file):
        error('The input file does not exist')
    opt.output_file = os.path.abspath(opt.output_file)

def get_feature_names():
    """Function to form the list of all the features with order conservation"""
    feature_names = list()
    for c in g_features.keys():
        feature_names.append(c)
    feature_names.append('diversity')
    feature_names.append('max_count')
    return np.array(feature_names)

#---Declaring functions---
def get_peptide_features(peptide):
    """Function for getting Numpy array with features of a peptide"""
    features = list()
    for c in g_features.values():
        features.append(len(tuple(c.finditer(peptide))))
    features_ = np.array(features[: 20])
    features.append(np.sum(features_ > 0)) #diversity
    features.append(np.max(features_)) #max_count
    return np.array(features)

#---Main section---
if __name__ == "__main__":
    features_matrix = []
    with open(opt.input_file, 'r') as ifile:
        for line in ifile:
            if not line.strip():
                continue
            features_matrix.append(get_peptide_features(line.strip()))
    features_matrix = np.array(features_matrix)
    np.save(opt.output_file, features_matrix)
    if opt.output_names:
        np.save(opt.output_names, get_feature_names())