import numpy as np
import sklearn.metrics as metric
import csv
import os
import sys
import logging
import traceback
import warnings


class DataSet:
    def __init__(self, data_set):
        """
        Initialize a data set and load both training and test data
        """
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        """
        Load a training set of the specified type (train/test). Returns None if either the training or test files were
        not found. NOTE: This is hard-coded to use only the first seven columns, and will not work with all data sets.
        """
        data_path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(data_path, missing_values=0, skip_header=0, delimiter=',', dtype=int)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(subs_path))


class Grader:
    def __init__(self, sid):
        print('Grading {0}'.format(sid))

        self.problems = ['balance-scale', 'breast-cancer', 'car', 'SPECT', 'tic-tac-toe']

        self.data = {}
        self.load_data_sets()

        self.utd_id = sid
        self.logger = None
        self.initialize_logging()

        # Import student-written functions to grade
        self.id3 = None
        self.predict_example = None
        self.import_functions()

        # Evaluate student-written functions to grade
        self.results = {}
        self.evaluate()

        # Generate a final report
        self.report = {}
        self.generate_report()

    # Load various data sets that will be used to evaluate each student's code
    def load_data_sets(self):
        for problem in self.problems:
            self.data[problem] = DataSet(problem)

    def initialize_logging(self):
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()

        # Remove all previous file handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler('./submissions/logs/{0}.log'.format(self.utd_id), mode='w')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

    # Import functions from student-submitted files dynamically
    def import_functions(self):
        self.logger.info('Attempting to import functions from {0}.py...'.format(self.utd_id))
        try:
            methods = __import__(self.utd_id, fromlist=['id3', 'predict_example'])
            self.id3 = getattr(methods, 'id3')
            self.predict_example = getattr(methods, 'predict_example')
        except Exception as e:
            self.logger.error('Could not dynamically import functions ''id3'' and ''predict_example''!')
            self.logger.error(e)

    # Evaluate functions from student-submitted files on several data sets
    def evaluate(self):
        # Suppress warnings
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

        for problem in self.problems:
            self.logger.info('Evaluating on data set: {0}'.format(problem))
            dimension = self.data[problem].dim                  # Get the problem dimension
            max_depth_allowed = np.min([dimension + 1, 11])     # Set max allowed depth to be 10 or problem dimension
            self.results[problem] = []                          # Initialize the results structure for this problem

            for max_depth in range(3, max_depth_allowed):
                self.logger.info('Trying to learn a tree with max_depth = {0}'.format(max_depth))
                attributes = np.array(range(dimension))  # Learn from all attributes

                # Learn a decision tree using the training data
                try:
                    tree = self.id3(self.data[problem].examples['train'],
                                    self.data[problem].labels['train'], attributes.tolist(), max_depth)
                    self.logger.info('Training... [OK]')
                except Exception:
                    self.logger.error('Training... [FAILED]')
                    # exc_type, exc_value, exc_traceback = sys.exc_info()
                    stack_trace = traceback.format_exc()
                    stack_trace = stack_trace.replace(
                                            'C:/Users/gxk170007/Dropbox/work-utd/teaching/CS6375-MachineLearning/', '.')
                    self.logger.error(stack_trace)
                    self.results[problem].append({'acc': 0.0, 'prc': 0.0, 'rec': 0.0})
                    continue

                # Predict the labels of the test data
                try:
                    y_pred = np.array([self.predict_example(self.data[problem].examples['test'][i, :], tree)
                                       for i in range(self.data[problem].num_test)])
                    self.logger.info('Testing... [OK]')
                except Exception:
                    self.logger.error('Testing... [FAILED]')
                    self.logger.error('This is most likely caused by a hard-coding of labels to 0 or 1.')
                    self.logger.error('NOTE: Other causes are possible, inspect the traceback below to debug.')
                    stack_trace = traceback.format_exc()
                    stack_trace = stack_trace.replace(
                                            'C:/Users/gxk170007/Dropbox/work-utd/teaching/CS6375-MachineLearning/', '.')
                    self.logger.error(stack_trace)

                    self.results[problem].append({'acc': 0.0, 'prc': 0.0, 'rec': 0.0})
                    continue

                # Compute some statistics on the model learned
                try:
                    y_pred = [0 if y is None or np.isnan(y) else y for y in y_pred]  # Convert all missing values to -1
                    y_true = self.data[problem].labels['test']  # Collect the true labels
                    results = {'acc': metric.accuracy_score(y_true, y_pred),
                               'prc': metric.precision_score(y_true, y_pred,
                                                             average='weighted', labels=np.unique(y_true)),
                               'rec': metric.recall_score(y_true, y_pred,
                                                          average='weighted', labels=np.unique(y_pred))}
                    self.results[problem].append(results)
                except Exception:
                    self.logger.error('Failed to generate results. This is most often caused by a failure in ')
                    self.logger.error('predict_example, specifically, when it returns a tree instead of a label.')
                    self.logger.error('NOTE: Other causes are possible, inspect the traceback below to debug.')
                    stack_trace = traceback.format_exc()
                    stack_trace = stack_trace.replace(
                                            'C:/Users/gxk170007/Dropbox/work-utd/teaching/CS6375-MachineLearning/', '.')
                    self.logger.error(stack_trace)

                    self.results[problem].append({'acc': 0.0, 'prc': 0.0, 'rec': 0.0})
                    continue

    def generate_report(self):
        report = ''
        passed = 0
        total = 0
        for problem, res_array in self.results.items():
            report += '\n-------------------------------------------  \n'
            report += '            Data Set: {0}               \n'.format(problem)
            report += '-------------------------------------------  \n\n'

            report += '  Depth |     Acc   |    Prec  |    Recl  \n'
            for d, result in enumerate(res_array):
                total += 1
                if result['acc'] > 0.15:
                    passed += 1
                report += '    {3:02d}  |   {0:5.4f}  |  {1:5.4f}  |  {2:5.4f} \n'.format(
                           result['acc'], result['prc'], result['rec'], d + 3)
            report += '\n'

        score = passed * 100.0 / total
        report += '\nFinal score = {0} / {1} ({2}).\n'.format(passed, total, score)

        self.report['text'] = report
        self.report['score'] = score
        self.logger.info('Generating report...' + report)


def load_student_files(rel_path):
    submissions = {}
    py_files = [g for g in os.listdir(rel_path) if '.py' in g]
    try:
        utd_ids = [file.split('.')[0] for file in py_files]  # Get the UTD IDs from the python file names
        py_files = [file for file in os.listdir(subs_path) if '.py' in file]  # Get only py files
        submissions = dict(zip(utd_ids, py_files))              # Make a dictionary of (utd_id, py_file)

    except Exception as e:
        print('Error loading assignment submissions from: {0}!'.format(subs_path))
        print('Current directory is: {0}.'.format(os.getcwd()))
        print(e)

    return submissions


# def rename_files(path):
#     try:
#         py_files = [f for f in os.listdir(path) if '.py' in f]  # Get only Python files from the directory
#         for sub in py_files:
#             uid = sub.split('_')[1]           # Get the UTD IDs from the python file names
#             os.rename(path + sub, path + '{0}.py'.format(uid))
#
#     except Exception as e:
#         print('Error renaming files!'.format(path))
#         print(e)


if __name__ == '__main__':
    # grader = Grader('./decision_tree_full')  # Gautam's code: will always run

    subs_path = os.path.abspath('./submissions/')  # Set path here
    sys.path.insert(0, subs_path)

    files_to_grade = load_student_files(subs_path)                       # Load all the files

    final_scores = {}
    for utd_id, submission in files_to_grade.items():
        if 'oxr170630' not in utd_id:
            continue

        grader = Grader(utd_id)                                 # Grade them one by one
        # report_file = '{0}/reports/{1}-report.txt'.format(subs_path, utd_id)     # Dump the report to a file
        # with open(report_file, 'w') as text_file:
        #     text_file.write(grader.report['text'])
        # final_scores[utd_id] = grader.report['score']               # Save the score

    # scores_file = '{0}/_scores.csv'.format(subs_path)                     # Dump the scores to a file
    # with open(scores_file, 'w') as f:
    #     w = csv.writer(f)
    #     w.writerows(final_scores.items())
