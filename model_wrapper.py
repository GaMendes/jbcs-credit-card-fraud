import shap
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, recall_score, precision_score,
                             f1_score, average_precision_score)
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from lime import lime_tabular
from sklearn.base import clone

class ModelWrapper:

    def __init__(self, model, params, random_seed):
        self.model              = model
        self.params             = params
        self.random_seed        = random_seed

        self.shap_explainers     = []
        self.lime_explainers     = []
        self.models              = []
        self.samples             = []
        self.precision_scores    = []
        self.f1_scores           = []
        self.balanced_accuracies = []
        self.sensitivity_scores  = []
        self.specificity_scores  = []
        self.auprc_scores        = []
        self.roc_auc_scores      = []

        self.training_time      = None
        self.optimization_time  = None
        self.shap_time          = None
        self.lime_time          = None

        np.random.seed(self.random_seed)

        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"ModelWrapper initialized for {self.model.__class__.__name__} with seed {self.random_seed}")

    def _optimize_hyperparameters(self, X, y):
        start = time.time()
        logging.info(f"Starting to calculate hyperparameters for {self.model.__class__.__name__}...")

        skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=self.random_seed)

        model_to_optimize = clone(self.model)
        if hasattr(model_to_optimize, 'random_state'):
             model_to_optimize.random_state = self.random_seed

        grid = GridSearchCV(model_to_optimize, self.params, cv=skf, n_jobs=-1)
        grid.fit(X, y)
        self.model = grid.best_estimator_

        end = time.time()
        self.optimization_time = end - start
        logging.info(f"Finished hyperparameter calculation. Time used: {self.optimization_time:.2f} seconds")
        logging.info(f"Best hyperparameters: {grid.best_params_}")

    def train(self, X, y, dataset):
        start = time.time()
        logging.info(f"Starting to train model {self.model.__class__.__name__}...")

        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=self.random_seed)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train = X_train.copy()
            X_test  = X_test.copy()

            if dataset == "kaggle":
                X_train, X_test = self.handle_scaling(X_train, X_test)

            if dataset == "ieee-cis":
                X_train, X_test = self.handle_numerical_imputation(X_train, X_test, ['TA', 'card1', 'card2', 'card3', 'card5'])
                X_train, X_test = self.handle_categorical_imputation(X_train, X_test, ['PCD', 'M1', 'M2', 'M3', 'M4', 'card4', 'card6', 'DT', 'DI', 'M5', 'M6', 'M7', 'M8', 'M9'])
                X_train, X_test = self.handle_catboost_encoding(X_train, X_test, y_train, y_test, ['PCD', 'M1', 'M2', 'M3', 'M4', 'card4', 'card6', 'DT', 'DI', 'M5', 'M6', 'M7', 'M8', 'M9'])
                X_train, X_test = self.handle_scaling(X_train, X_test)

            if dataset == "synthetic":
                X_train, X_test = self.handle_numerical_imputation(X_train, X_test, ['Year', 'Month', 'Day', 'Amount', 'MCC'])
                X_train, X_test = self.handle_categorical_imputation(X_train, X_test, ['UC', 'MN', 'MC', 'MS', 'Zip', 'errors'])
                X_train, X_test = self.handle_catboost_encoding(X_train, X_test, y_train, y_test, ['UC', 'MN', 'MC', 'MS', 'Zip', 'errors'])
                X_train, X_test = self.handle_scaling(X_train, X_test)

            # Store a random positive sample
            positive_samples_index = np.where(y_train == 1)[0]
            random_positive_sample_index = np.random.choice(positive_samples_index)
            random_positive_sample = X_train.iloc[random_positive_sample_index]
            self.samples.append(random_positive_sample)

            # Oversampling with SMOTE
            oversample = SMOTE(random_state=self.random_seed)
            X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)

            # Model training
            self._optimize_hyperparameters(X_train_smote, y_train_smote)
            self.model.fit(X_train_smote, y_train_smote)
            self.models.append(self.model)

            # SHAP Explainer
            self.explain_shap(X_train_smote)

            # LIME Explainer
            self.explain_lime(X_train_smote)

            # Calculate and store metrics
            y_pred = self.model.predict(X_test)
            self.balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            self.sensitivity_scores.append(recall_score(y_test, y_pred))
            self.specificity_scores.append(recall_score(y_test, y_pred, pos_label=0))
            self.precision_scores.append(precision_score(y_test, y_pred))
            self.f1_scores.append(f1_score(y_test, y_pred))
            self.auprc_scores.append(average_precision_score(y_test, y_pred))

        end = time.time()
        self.training_time = end - start
        logging.info(f"Finished training model {self.model.__class__.__name__}. Time used: {self.training_time:.2f} seconds")

        return self.get_results()

    def explain_shap(self, X):
        start = time.time()
        logging.info("Starting to create SHAP explainer...")

        shap_explainer = shap.Explainer(self.model.predict, X, seed=self.random_seed)
        self.shap_explainers.append(shap_explainer)

        end = time.time()
        self.shap_time = end - start
        logging.info(f"Finished SHAP explainer creation. Time used: {self.shap_time:.2f} seconds")

    def explain_lime(self, X):
        start = time.time()
        logging.info("Starting to create LIME explainers...")

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=X.columns,
            class_names=['legítima', 'fraude'],
            mode='classification',
            random_state=self.random_seed)
        self.lime_explainers.append(lime_explainer)

        end = time.time()
        self.lime_time = end - start
        logging.info(f"Finished to create LIME exaplainers. Time used: {self.lime_time:.2f} seconds")

    def handle_scaling(self, X_train, X_test):
        start = time.time()
        logging.info(f"Starting to scale data...")
        
        X = X_train.copy()
        scaler  = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        end = time.time()
        scaling_time = end - start
        logging.info(f"Finished scaling data. Time used: {scaling_time:.2f} seconds")

        return pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(X_test, columns=X.columns)

    def handle_numerical_imputation(self, X_train, X_test, cols):
        start = time.time()
        logging.info(f"Start numerical imputation...")

        numeric_cols = cols
        numeric_imputer = SimpleImputer(strategy='median')
        X_train.loc[:, numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_test.loc[:, numeric_cols]  = numeric_imputer.transform(X_test[numeric_cols])

        end = time.time()
        imputation_time = end - start
        logging.info(f"Finished numerical imputation. Time used: {imputation_time:.2f} seconds")

        return pd.DataFrame(X_train), pd.DataFrame(X_test)

    def handle_categorical_imputation(self, X_train, X_test, cols):
        start = time.time()
        logging.info(f"Start categorical imputation...")

        categorical_cols = cols
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_train.loc[:, categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_test.loc[:, categorical_cols]  = categorical_imputer.transform(X_test[categorical_cols])

        end = time.time()
        imputation_time = end - start
        logging.info(f"Finished categorical imputation. Time used: {imputation_time:.2f} seconds")

        return pd.DataFrame(X_train), pd.DataFrame(X_test)

    def handle_catboost_encoding(self, X_train, X_test, y_train, y_test, cols):
        start = time.time()
        logging.info(f"Start catboost encoding...")

        encoder = CatBoostEncoder(random_state=self.random_seed)
        categorical_cols = cols
        encoded_cols = encoder.fit_transform(X_train.loc[:, categorical_cols], y_train)
        X_train = pd.concat([X_train.drop(columns=categorical_cols), encoded_cols], axis=1)
        encoded_cols = encoder.transform(X_test.loc[:, categorical_cols], y_test)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), encoded_cols], axis=1)

        end = time.time()
        encoding_time = end - start
        logging.info(f"Finished catboost encoding. Time used: {encoding_time:.2f} seconds")

        return pd.DataFrame(X_train), pd.DataFrame(X_test)

    def get_results(self):

        """
        Returns the results of the training process.
        """

        return {
            'shap_explainers': self.shap_explainers,
            'lime_explainers': self.lime_explainers,
            'models': self.models,
            'samples': self.samples,
            'sensitivity_scores': self.sensitivity_scores,
            'balanced_accuracies': self.balanced_accuracies,
            'specificity_scores': self.specificity_scores,
            'precision_scores': self.precision_scores,
            'f1_scores': self.f1_scores,
            'auprc_scores': self.auprc_scores,
            'training_time': self.training_time,
            'optimization_time': self.optimization_time,
            'shap_time': self.shap_time,
            'lime_time': self.lime_time
        }