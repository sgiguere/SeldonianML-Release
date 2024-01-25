import numpy as np
import warnings
from time import time
import pandas as pd

from sklearn import linear_model as LM
from sklearn.svm import SVC, LinearSVC

from baselines.fairlearn.classred import expgrad
from baselines.fairlearn import moments
import baselines.fair_classification.utils as fc_ut
import baselines.fair_classification.loss_funcs as fc_lf

# SeldonianML imports
from utils             import argsweep
from utils.experiments import launcher
from datasets          import propublica as ProPublica
from core.sc_fairness  import get_sc_class

# Supress sklearn FutureWarnings for SGD
warnings.simplefilter(action='ignore', category=FutureWarning)


########################
#   Model Evaluators   #
########################

def eval_fairlearn(dataset, mp):
	# Train the model
	t = time()
	# Load the dataset and convert it to a pandas dataframe
	Xt, Yt, Tt = dataset.training_splits()
	Xt = pd.DataFrame(Xt)
	# Convert Y to be in {0,1} instead of {-1,1} for compatibility with fairlearn
	Yt[Yt==-1] = 0
	Yt = pd.Series(Yt)
	Tt = pd.Series(Tt)
	# Use expgrad with a linear SVC

	# Note that this fairlearn implementation only supports DemographicParity and EqualOpportunity
	# When other definitions are requested, we enforce DP or EO based on which is most reasonable
	defs = {
		'demographicparity'  : moments.DP,
		'disparateimpact'    : moments.EO,
		'equalizedodds'      : moments.EO,
		'equalopportunity'   : moments.EO,
		'predictiveequality' : moments.EO }
	cons = defs[mp['definition'].lower()]()
	
	# Train fairlearn using expgrad with a linear SVC
	base_model = LinearSVC(loss=mp['loss'], penalty=mp['penalty'], fit_intercept=mp['fit_intercept'])
	results = expgrad(Xt, Tt, Yt, base_model, cons=cons, eps=mp['fl_e'])

	# Evaluate the model
	t_train = time() - t
	def predictf(X):
		Yp = results.best_classifier(X)
		try:
			Yp[Yp==0] = -1
		except TypeError:
			Yp = Yp if Yp == 1 else -1
		return Yp
	SC_Class = get_sc_class(mp['definition'], 'ttest')
	results = SC_Class(epsilon=mp['e'], shape_error=True).evaluate(dataset, predictf)
	results['train_time'] = t_train
	return results

def eval_fair_constraints(dataset, mp):
	# FairConstraints is constructed to simultaneously enforce disparate impact and disparate treatment,
	# thus the training process is the same regardless of the actual definition we're evaluating.
	t = time()
	# Configure the constraints and weights
	apply_fairness_constraints = 1
	apply_accuracy_constraint  = 0
	sep_constraint = 0
	gamma = None
	e = -mp['e']*100
	# Train the model using the cov that produced the smallest p >= e
	X, Y, T = dataset.training_splits()
	sensitive_attrs_to_cov_thresh = {'T':mp['cov']}
	w = fc_ut.train_model(X, Y, {'T':T.astype(np.int64)}, fc_lf._logistic_loss, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['T'], sensitive_attrs_to_cov_thresh, gamma)
	t_train = time() - t
	def predictf(_X):
		Yp = np.sign(np.dot(_X, w))
		try:
			Yp[Yp==0] = -1
		except:
			pass
		return Yp
	# Evaluate the model
	SC_Class = get_sc_class(mp['definition'], 'ttest')
	results = SC_Class(epsilon=mp['e'], shape_error=True).evaluate(dataset, predictf)
	results['train_time'] = t_train
	return results

def eval_hoeff_sc(dataset, mp):
	model_params = {
		'epsilon' : mp['e'],
		'delta'   : mp['d'],
		'verbose' : False,
		'shape_error' : True,
		'model_type'  : mp['model_type'] }
	# Train the model
	t = time()
	SC_Class = get_sc_class(mp['definition'], 'Hoeffding')
	model = SC_Class(**model_params)
	model.fit(dataset, n_iters=mp['n_iters'], optimizer_name=mp['optimizer'])
	t_train = time() - t
	# Assess the model
	results = model.evaluate(dataset)
	results['train_time'] = t_train
	return results

def eval_ttest_sc(dataset, mp):
	model_params = {
		'epsilon' : mp['e'],
		'delta'   : mp['d'],
		'verbose' : False,
		'shape_error' : True,
		'model_type'  : mp['model_type'] }
	# Train the model
	t = time()
	SC_Class = get_sc_class(mp['definition'], 'ttest')
	model = SC_Class(**model_params)
	model.fit(dataset, n_iters=mp['n_iters'], optimizer_name=mp['optimizer'])
	t_train = time() - t
	# Assess the model
	results = model.evaluate(dataset)
	results['train_time'] = t_train
	return results

def eval_sgd(dataset, mp):
	# Train the model
	t = time()
	Xt, Yt, _ = dataset.training_splits()
	if mp['loss']=='log':
		model = LM.LogisticRegression(fit_intercept=mp['fit_intercept'])
	else:
		model = LM.SGDClassifier(loss=mp['loss'], penalty=mp['penalty'], fit_intercept=mp['fit_intercept'], max_iter=1000, alpha=0.000001)
	model.fit(Xt, Yt)
	t_train = time() - t
	# Evaluate the model
	SC_Class = get_sc_class(mp['definition'], 'Hoeffding')
	results = SC_Class(epsilon=mp['e'], shape_error=True).evaluate(dataset, model.predict)	
	results['train_time'] = t_train
	return results

def eval_svc(dataset, mp):
	# Train the model
	t = time()
	Xt, Yt, _ = dataset.training_splits()
	model = SVC(gamma=mp['gamma'], C=mp['C'], kernel=mp['kernel'])
	model.fit(Xt, Yt)
	t_train = time() - t
	# Evaluate the model
	SC_Class = get_sc_class(mp['definition'], 'Hoeffding')
	results = SC_Class(epsilon=mp['e'], shape_error=True).evaluate(dataset, model.predict)	
	results['train_time'] = t_train
	return results

def eval_linsvc(dataset, mp):
	# Train the model
	t = time()
	Xt, Yt, _ = dataset.training_splits()
	model = LinearSVC(loss=mp['loss'], penalty=mp['penalty'], fit_intercept=mp['fit_intercept'])
	model.fit(Xt, Yt)
	t_train = time() - t
	# Evaluate the model
	SC_Class = get_sc_class(mp['definition'], 'Hoeffding')
	results = SC_Class(epsilon=mp['e'], shape_error=True).evaluate(dataset, model.predict)
	results['train_time'] = t_train
	return results


######################
#   Dataset Loader   #
######################

def load_dataset(tparams, seed):
	dset_args = {
		'r_train'     : tparams['r_train_v_test'], 
		'r_candidate' : tparams['r_cand_v_safe'], 
		'T0'          : tparams['T0_label'], 
		'T1'          : tparams['T1_label'], 
		'dset_type'   : tparams['dset_type'], 
		'include_T'   : tparams['include_T'], 
		'use_pct'     : tparams['data_pct'], 
		'include_intercept' : True,
		'standardize' : tparams['standardize'],
		'seed'        : seed 
	}
	return ProPublica.load(**dset_args)	


############
#   Main   #
############

if __name__ == '__main__':

	# Note: This script computes experiments for the cross product of all values given for the
	#       sweepable arguments. 
	# Note: Sweepable arguments allow inputs of the form, <start>:<end>:<increment>, which are then
	#       expanded into ranges via np.arange(<start>, <end>, <increment>). 
	with argsweep.ArgumentSweeper() as parser:
		parser.add_argument('base_path', type=str)
		parser.add_argument('--T0_label',   type=str,  default='African-American', help='ID for type 0.')
		parser.add_argument('--T1_label',   type=str,  default='Caucasian',        help='ID for type 1.')
		parser.add_argument('--dset_type',  type=str,  default='violent',          help='Version of the ProPublica data to use (violent or nonviolent).')
		parser.add_argument('--include_T',  action='store_true', help='Whether or not to include type as a predictive feature.')
		parser.add_argument('--standardize',  action='store_true', help='Whether or not to standardize input features.')
		parser.add_argument('--n_jobs',     type=int,  default=4,     help='Number of processes to use.')
		parser.add_argument('--n_trials',   type=int,  default=10,    help='Number of trials to run.')
		parser.add_argument('--n_iters',    type=int,  default=10,    help='Number of SMLA training iterations.')
		parser.add_argument('--optimizer',         type=str,   default='cmaes', help='Choice of optimizer to use.')
		parser.add_argument('--definition', type=str,   default='DisparateImpact',        help='Choice of safety definition to enforce.')
		parser.add_sweepable_argument('--data_pct',   type=float,  default=1.0,   nargs='*', help='Percentage of the overall size of the dataset to use.')
		parser.add_sweepable_argument('--e',          type=float,  default=0.05,  nargs='*', help='Values for epsilon.')
		parser.add_sweepable_argument('--d',          type=float,  default=0.05,  nargs='*', help='Values for delta.')
		parser.add_sweepable_argument('--r_train_v_test', type=float, default=0.4,  nargs='*', help='Ratio of data used for training vs testing.')
		parser.add_sweepable_argument('--r_cand_v_safe',  type=float, default=0.4,  nargs='*', help='Ratio of training data used for candidate selection vs safety checking. (SMLA only)')
		parser.add_sweepable_argument('--model_type',     type=str, default='linear', nargs='*', help='Base model type to use for SMLAs.')
		args = parser.parse_args()
		args_dict = dict(args.__dict__)
		
		# Resolve the names for the SMLAs that will be tested
		model_name_h = get_sc_class(args.definition, 'hoeffding').__name__
		model_name_t = get_sc_class(args.definition, 'ttest').__name__
		smla_names = [model_name_h, model_name_t]

		model_evaluators = {
			model_name_h   : eval_hoeff_sc,
			model_name_t   : eval_ttest_sc,
			'SGD'          : eval_sgd,
			'LinSVC'       : eval_linsvc,
			'SVC'          : eval_svc,
			'FairConst'    : eval_fair_constraints,
			'FairlearnSVC' : eval_fairlearn
		}

		#    Store task parameters:
		tparams = {k:args_dict[k] for k in ['n_jobs', 'base_path', 'data_pct', 'T0_label', 'T1_label', 'r_train_v_test', 'r_cand_v_safe', 'include_T', 'dset_type', 'standardize']}
		#    Store method parameters:

		srl_mparam_names  = ['e','d','n_iters','optimizer','model_type', 'definition']
		bsln_mparam_names = ['e', 'd', 'definition']
		mparams = {}
		for name in model_evaluators.keys():
			if name in smla_names:
				mparams[name] = {k:args_dict[k] for k in srl_mparam_names}
			else:
				mparams[name] = {k:args_dict[k] for k in bsln_mparam_names}
		mparams['SGD'].update(loss=['hinge','log','perceptron'], penalty='l2', fit_intercept=False)
		mparams['SVC'].update(kernel=['rbf'], gamma=2, C=1)
		mparams['LinSVC'].update(loss=['hinge'], penalty='l2', fit_intercept=False)
		mparams['FairConst'].update(cov=[0.01, 0.1, 1.0])
		mparams['FairlearnSVC'].update(loss=['hinge'], penalty='l2', fit_intercept=False, fl_e=[0.01, 0.1, 1.0])

		#    Expand the parameter sets into a set of configurations
		args_to_expand = parser._sweep_argnames + ['loss', 'kernel', 'cov', 'fl_e']
		tparams, mparams = launcher.make_parameters(tparams, mparams, expand=args_to_expand)
		print()
		# Create a results file and directory
		save_path = launcher.prepare_paths(args.base_path, tparams, mparams, smla_names, root='results', filename=None)
		print()
		# Run the experiment
		launcher.run(args.n_trials, save_path, model_evaluators, load_dataset, tparams, mparams, n_workers=args.n_jobs, seed=None)