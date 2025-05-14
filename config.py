from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_datasets_to_run():

    """Returns a list of dataset names to process."""

    return ['kaggle',
            'ieee-cis', 
            'synthetic'
            ]

def get_models_to_run():

    """Returns a dictionary of model names and their instances."""

    return {
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'LinearSVC': LinearSVC(),
        'MLP': MLPClassifier()
    }

def get_model_params(model_name, seed):

    """Returns hyperparameters for grid search for a given model name."""

    if model_name == 'DecisionTree':
        return {'max_depth': [None, 5, 10], 'random_state': [seed]}

    if model_name == 'RandomForest':
        return {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10], 'random_state': [seed]}

    if model_name == 'LogisticRegression':
        return {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver':['newton-cholesky'], 'max_iter':[2000, 5000], 'n_jobs': [-1], 'random_state': [seed]}

    if model_name == 'LinearSVC':
        return {'C': [0.1, 1, 10], 'dual':[False], 'random_state': [seed]}

    if model_name == 'MLP':
        return {'hidden_layer_sizes': [(100,), (50, 50)], 'max_iter': [200, 400], 'random_state': [seed]}

    return {}