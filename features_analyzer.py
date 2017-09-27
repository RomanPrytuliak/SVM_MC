# The script containing the SVM-based machine learning pipeline that develops a model based on the provided training set and than chacks the model performance on the OOB (out-of-bag) dataset.
# There are three levels of training-test partitions in the pipeline. The outmost test set is provided in separate files and is labeled 'OOB'. The performance on this dataset is labelled 'OOB score'.
# Parameters C (SVM slack) and k (the number of features to keep) are optimized at the innermost partition in a CV procedure. The performance on this dataset is labelled 'CV score'.
# The middle layer is introduced to control for possible overfitting at the innermost level. The performance on this dataset is labelled 'test score'.
# The overall pipeline has overfitting controls at several points ensuring that the outputted performance on the OOB dataset is a reliable value

import os, sys, argparse, sharedmem, multiprocessing, functools
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
from sklearn import svm, model_selection, preprocessing, feature_selection, pipeline, metrics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import error

#Processing command line arguments
usage = "Usage: %(prog)s [options] -p POSITIVE_SET_FILE -n NEGATIVE_SET_FILE"
parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('-ip', dest = 'positive_set_file', required = True, help = 'Input numpy dump file with positive dataset')
parser.add_argument('-in', dest = 'negative_set_file', required = True, help = 'Input numpy dump file with negative dataset')
parser.add_argument('-oob_p', dest = 'oob_positive_set_file', required = True, help = 'Input numpy dump file with positive OOB dataset')
parser.add_argument('-oob_n', dest = 'oob_negative_set_file', required = True, help = 'Input numpy dump file with negative OOB dataset')
parser.add_argument('-names', dest = 'feature_names_file', default = '', help = 'Input gzip file with feature names')
if __name__ == "__main__":
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    opt = parser.parse_args()
    opt.positive_set_file = os.path.abspath(opt.positive_set_file)
    if not os.path.isfile(opt.positive_set_file):
        error('The positive set file does not exist')
    opt.negative_set_file = os.path.abspath(opt.negative_set_file)
    if not os.path.isfile(opt.negative_set_file):
        error('The negative set file does not exist')
    if not os.path.isfile(opt.oob_positive_set_file):
        error('The positive OOB set file does not exist')
    opt.negative_set_file = os.path.abspath(opt.negative_set_file)
    if not os.path.isfile(opt.oob_negative_set_file):
        error('The negative OOB set file does not exist')
    if opt.feature_names_file:
        opt.feature_names_file = os.path.abspath(opt.feature_names_file)

def worker(selection_idx, results_table):
    """The worker function to run an inner CV round. Makes a single"""
    randgen = np.random.RandomState()
    
    # Data-specific positive set partition (the real-world dataset consists of multiple motif classes, always exactly 3 instances of each class stored consequently).
    # The partition assures that the training and test sets do not share instances of the same motif class
    positive_n_train = round(0.8 * len(positive_set_) / 3) * 3
    block_start_idx = randgen.randint(positive_n_train / 3 + 1) * 3 
    block_end_idx = block_start_idx + len(positive_set_) - positive_n_train
    positive_set_part_train, positive_set_part_test = (np.concatenate((positive_set_[: block_start_idx], positive_set_[block_end_idx: ])), positive_set_[block_start_idx: block_end_idx])
    
    # Negative set partition with random selection of elements to match the size of the positive set
    negative_set = negative_set_[randgen.choice(len(negative_set_), size = positive_set_.shape[0], replace = False)]
    negative_n = len(negative_set)
    negative_n_train = round(negative_n * 0.8)
    negative_set_part_train, negative_set_part_test = (negative_set[: negative_n_train], negative_set[negative_n_train: ])
    
    data_part_train = np.float64(np.concatenate((positive_set_part_train, negative_set_part_train)))
    labels_part_train = np.concatenate((np.ones(len(positive_set_part_train), dtype = 'i1'), np.zeros(len(negative_set_part_train), dtype = 'i1')))
    data_part_test = np.float64(np.concatenate((positive_set_part_test, negative_set_part_test)))
    labels_part_test = np.concatenate((np.ones(len(positive_set_part_test), dtype = 'i1'), np.zeros(len(negative_set_part_test), dtype = 'i1')))
    
    # Specifying the pipeline and the CV structure
    pruner = feature_selection.VarianceThreshold()
    scaler = preprocessing.StandardScaler()
    feature_selector = feature_selection.SelectKBest(feature_selection.f_classif)
    classifier = svm.SVC(kernel = 'rbf', gamma = 0.01, class_weight = 'balanced')
    pipeline0 = pipeline.Pipeline([
        ('pruning', pruner),
        ('scaling', scaler),
        ('selection', feature_selector),
        ('classification', classifier)
    ])
    cv_structure = model_selection.StratifiedShuffleSplit(n_splits = 10, test_size = 0.2)
    scoring = 'recall_macro' #same as balanced accuracy
    grid = model_selection.GridSearchCV(pipeline0, scoring = scoring, param_grid = param_grid, cv = cv_structure, n_jobs = 1)
    
    # Training the pipeline, saving the data
    grid.fit(data_part_train, labels_part_train)
    results_table[selection_idx][0] = np.log10(grid.best_params_['classification__C'])
    results_table[selection_idx][1] = grid.best_params_['selection__k']
    results_table[selection_idx][2] = grid.best_score_
    
    # Testing the pipeline, saving the data
    results_table[selection_idx][3] = grid.score(data_part_test, labels_part_test)

# Loading input data
positive_set_ = np.float64(np.load(opt.positive_set_file))
negative_set_ = np.float64(np.load(opt.negative_set_file))
feature_names = np.load(opt.feature_names_file) if opt.feature_names_file else np.zeros(positive_set_.shape[1], dtype = '<U1')
features_n = feature_names.shape[0]
oob_positive_ =  np.float64(np.load(opt.oob_positive_set_file))
oob_negative_ =  np.float64(np.load(opt.oob_negative_set_file))

# Filtering the data, removing duplicates from the negative sets, assuring that the OOB negative dataset does not contain elements present in the training set
# The positive sets are not controlled for duplicates to preserve the 3-blocks sorted by motif classes. Duplicates are extremely unlikely due to relatively small dataset sizes. 
negative_set_ = np.unique(negative_set_, axis = 0)
negative_set_records = np.core.records.fromarrays(negative_set_.T, formats = ', '.join(['f8'] * features_n))
oob_negative_records = np.core.records.fromarrays(oob_negative_.T, formats = ', '.join(['f8'] * features_n))
oob_negative_unique = np.setdiff1d(oob_negative_records, negative_set_records)
del negative_set_records, oob_negative_records
oob_negative_ = oob_negative_unique.view(np.float64).reshape(oob_negative_unique.shape[0], features_n)
del oob_negative_unique
oob_data = np.concatenate((oob_positive_, oob_negative_))
oob_labels = np.concatenate((np.ones(oob_positive_.shape[0]), np.zeros(oob_negative_.shape[0])))

# Setting the parameters
n_selections = 25 #number of random training set partitions
n_repeats = 2##/ #number of repeats of the overall procedure
param_grid = dict()
param_grid['classification__C'] = np.logspace(-3, 3, 13)
param_grid['selection__k'] = np.linspace(20, 400, 20, dtype = 'i2')
score_diff_alpha = 0.01 #the lowest allowed p-value for the difference between scores on different test set layers
C_SD_max = 1.0 #maximal allowed SD of optimal C in the logarithmic scale
k_SD_max = 100.0 #maximal allowed SD of optimal k

oob_scores = np.zeros(n_repeats)
for outer_run in range(n_repeats):
    
    # Optimizing parameters, performing cross-validation on the training dataset
    results_table = sharedmem.empty((n_selections, 4)) #contains teh best C, best k, outer CV score and test score for each training set partition
    pool = multiprocessing.Pool(processes = 40)
    pool.map(functools.partial(worker, results_table = results_table), range(n_selections))
    pool.close()
    results_means = np.mean(results_table, axis = 0)
    
    # Overfitting control. Discard the model if the variance in the optimal parameters or the difference between CV and test scores is too large
    C_SD = np.std(results_table[:, 0])
    k_SD = np.std(results_table[:, 1])
    p_value = ttest_ind(results_table[:, 2], results_table[:, 3], equal_var = False)[1]
    if (C_SD > C_SD_max) or (k_SD > k_SD_max) or (p_value < score_diff_alpha):
        oob_scores[outer_run] = np.nan
        continue
    
    # Assembling the final pipeline trained on the whole training positive dataset with the optimized parameters
    pruner = feature_selection.VarianceThreshold()
    scaler = preprocessing.StandardScaler()
    feature_selector = feature_selection.SelectKBest(feature_selection.f_classif , k = np.round(results_means[1]).astype(np.int32))
    classifier = svm.SVC(kernel = 'rbf', C = np.power(10, results_means[0]), gamma = 0.01, class_weight = 'balanced')
    pipeline_final = pipeline.Pipeline([
        ('pruning', pruner),
        ('scaling', scaler),
        ('selection', feature_selector),
        ('classification', classifier)
    ])
    randgen = np.random.RandomState()
    negative_set = negative_set_[randgen.choice(negative_set_.shape[0], size = positive_set_.shape[0], replace = False)]
    data = np.float64(np.concatenate((positive_set_, negative_set)))
    labels = np.concatenate((np.ones(positive_set_.shape[0], dtype = 'i1'), np.zeros(positive_set_.shape[0], dtype = 'i1')))
    pipeline_final.fit(data, labels)
    
    # Validating the pipeline on the OOB dataset
    oob_labels_pred = pipeline_final.predict(oob_data)
    oob_score = metrics.recall_score(oob_labels, oob_labels_pred, average = 'macro')
    
    # Overfitting control. Discard the OOB score if it deviates from the test scores too much
    p_value = ttest_1samp(results_table[:, 3], oob_score)[1]
    if p_value < score_diff_alpha:
        oob_scores[outer_run] = np.nan
        continue
    
    oob_scores[outer_run] = oob_score

print('--- OOB test summary: ---')
print('total runs: {}'.format(n_repeats))
n_failed = np.sum(np.isnan(oob_scores))
print('failed: {}'.format(n_failed))
n_passed = n_repeats - n_failed
if n_passed:
    print('passed: {} with the mean {:.3f}{}'.format(n_passed, np.nanmean(oob_scores), ' and SD {:.3f}'.format(np.nanstd(oob_scores)) if n_passed > 1 else ''))