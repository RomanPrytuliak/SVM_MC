# This script runs the other scripts to generate random peptide datasets with desirable fuzzy pattren distributions with subsequent test of the ability of the SVM-based machine learning pipeline to detect the differences
# between non-equivalent distributions

import os
from subprocess import Popen
from thesis_constants import MC_patterns

# 100K negative samples and 250 positive ones - close to the real-world scenario
negative_count = str(100000)
positive_count = str(250)

prefixes = ['train_', 'oob_']
base = 'MC_set'
suffixes = ['_negative', '_positive']
feature_names_filename = 'feature_names.npy'
dir_ = os.path.dirname(__file__) + '/'

for MC_series in MC_patterns.keys():
    
    # Generating the random peptides - a pair of 'training' and a pair of 'OOB' (out-of-bag) datasets
    for prefix in prefixes:
        Popen(['python3', dir_ + 'random_patterned_peptide_generator.py', '-on', prefix + base + suffixes[0] + '.txt', '-op', prefix + base + suffixes[1] + '.txt', '-nn', negative_count, '-np', positive_count, '-s',
               MC_series]).communicate()
        for suffix in suffixes:
            Popen(['python3', dir_ + 'peptide_features_generator.py', '-i', prefix + base + suffix + '.txt', '-o', prefix + base + suffix + '.npy', '-names', feature_names_filename]).communicate()
    print('Monte Carlo Series {}: Sets and features are generated'.format(MC_series))
    
    # Testing the pipeline
    Popen(['python3', dir_ + 'features_analyzer.py', '-in', 'train_MC_set_negative.npy', '-ip', 'train_MC_set_positive.npy', '-oob_n', 'oob_MC_set_negative.npy', '-oob_p', 'oob_MC_set_positive.npy', '-names',
           feature_names_filename]).communicate()