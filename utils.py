import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
import sklearn.svm

RESULTS_BASE_DIR = Path("./results")
SUMMARY_BASE_DIR = RESULTS_BASE_DIR / "summary"

def create_results_dir(experiment_name, dataset_name, model_name):

    results_dir = RESULTS_BASE_DIR / experiment_name / dataset_name / model_name

    try:

        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results directory created/verified: {results_dir}")

    except OSError as e:

        logging.error(f"Failed to create results directory {results_dir}: {e}")
        raise

    return results_dir

def create_summary_dir(dataset_name, model_name):

    summary_dir = SUMMARY_BASE_DIR / dataset_name / model_name

    try:
        summary_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Summary directory created/verified: {summary_dir}")
  
    except OSError as e:

        logging.error(f"Failed to create summary directory {summary_dir}: {e}")
        raise

    return summary_dir

class ResultsSaver:

    def __init__(self, results_dict, seed, results_dir, model_name):

        """
        Initializes the saver with the results from ModelWrapper.train().

        Args:
            results_dict (dict): The dictionary returned by ModelWrapper.get_results().
            seed (int): The random seed used for this experiment run.
            results_dir (Path): The directory to save the files for this run.
        """

        self.results = results_dict
        self.seed = seed
        self.results_dir = results_dir
        self.model_name = model_name
        self.feature_names = results_dict.get('samples', [])[0].keys()
        self.num_folds = len(results_dict.get('models', []))

    def save_metrics(self):

        """Saves evaluation metrics to metrics.csv."""

        balanced_acc_scores = self.results.get('balanced_accuracies', [])
        if not balanced_acc_scores:
            logging.warning(f"Seed {self.seed}, Model {self.model_name}: No balanced accuracy scores found. Skipping metrics saving.")
            return pd.DataFrame()

        num_scores = len(balanced_acc_scores)
        if num_scores != self.num_folds:
            logging.warning(f"Seed {self.seed}, Model {self.model_name}: Mismatch between number of folds ({self.num_folds}) and number of balanced accuracy scores ({num_scores}). Metrics might be incomplete.")


        try:
            data = {
                'fold': list(range(1, num_scores + 1)),
                'balanced_accuracy': balanced_acc_scores,
                'sensitivity_recall': self.results.get('sensitivity_scores', [np.nan] * num_scores),
                'specificity': self.results.get('specificity_scores', [np.nan] * num_scores),
                'precision': self.results.get('precision_scores', [np.nan] * num_scores),
                'f1': self.results.get('f1_scores', [np.nan] * num_scores),
                'auprc': self.results.get('auprc_scores', [np.nan] * num_scores),
            }

            metrics_df = pd.DataFrame(data)
            metrics_df['seed'] = self.seed

            cols_order = ['seed', 'fold', 'balanced_accuracy', 'sensitivity_recall', 'specificity', 'precision', 'f1', 'auprc', 'roc_auc']
            metrics_df = metrics_df[[col for col in cols_order if col in metrics_df.columns]]

            filepath = self.results_dir / "metrics.csv"
            metrics_df.to_csv(filepath, index=False, float_format='%.6f')

            logging.info(f"Seed {self.seed}, Model {self.model_name}: Saved metrics to {filepath}")

            return metrics_df

        except Exception as e:
            logging.error(f"Seed {self.seed}, Model {self.model_name}: Error processing or saving metrics: {e}", exc_info=True)
            return pd.DataFrame()

    def save_timings(self):

        """Saves detailed step timings to timings.csv."""

        times_data = self.results
        if not times_data:
            logging.warning(f"Seed {self.seed}: No timing data found to save.")
            return pd.DataFrame()

        data = {
            'training_time': times_data['training_time'],
            'optimization_time': times_data['optimization_time'],
            'shap_time': times_data['shap_time'],
            'lime_time': times_data['lime_time']
        }

        try:

            if isinstance(times_data, pd.DataFrame):
                timing_df = times_data.copy()
            else:
                timing_df = pd.DataFrame(data, index=[0])

            timing_df['seed'] = self.seed

            rename_map = {
                'optimization_time':'optimization_time',
                'training_time':'training_time',
                'shap_time':'shap_explainer_creation_time',
                'lime_time':'lime_explainer_creation_time'
            }

            timing_df = timing_df.rename(columns=rename_map)

            cols_order = ['seed', 'fold', 'preprocessing_time', 'smote_time', 'optimization_time', 'training_time', 'shap_explainer_creation_time', 'lime_explainer_creation_time']
            timing_df = timing_df[[col for col in cols_order if col in timing_df.columns]]

            filepath = self.results_dir / "timings.csv"
            timing_df.to_csv(filepath, index=False, float_format='%.4f')

            logging.info(f"Seed {self.seed}: Saved detailed timing information to {filepath}")
            return timing_df

        except Exception as e:
            logging.error(f"Seed {self.seed}: Error processing or saving timing details: {e}", exc_info=True)
            return pd.DataFrame()

    def _generate_lime_explanations(self):

        """Generates LIME explanation values using stored explainers and samples."""

        lime_explainers = self.results.get('lime_explainers', [])
        samples = self.results.get('samples', [])
        models =  self.results.get('models', [])
        all_lime_results = []
        total_lime_gen_time = 0

        logging.info(f"Seed {self.seed}: Generating LIME explanation values...")

        for i in range(self.num_folds):
            fold_num = i + 1
            lime_explainer = lime_explainers[i] if i < len(lime_explainers) else None
            fraud_sample = samples[i] if i < len(samples) else None
            model = models[i]

            if lime_explainer is None or fraud_sample is None or fraud_sample.empty:
                logging.warning(f"Fold {fold_num}, Seed {self.seed}: LIME explainer or sample missing, skipping explanation.")
                continue

            lime_start_time = time.time()
            explanation_list = None

            def lime_predict_fn_wrapper(data_as_numpy_array):

              if isinstance(model, (sklearn.svm.LinearSVC, sklearn.svm.SVC)) and hasattr(model, '_predict_proba_lr'):
                  return model._predict_proba_lr(data_as_numpy_array)

              elif hasattr(model, 'predict_proba'):
                  return model.predict_proba(data_as_numpy_array)

              else:
                  raise TypeError(f"Model of type {type(model)} accessible in scope does not have "
                                  f"a suitable method (_predict_proba_lr or predict_proba) "
                                  f"for LIME explanations.")

            try:
                explanation = lime_explainer.explain_instance(
                    data_row=fraud_sample,
                    predict_fn=lime_predict_fn_wrapper
                )

                explanation_list = explanation.as_list()
                lime_time = time.time() - lime_start_time
                total_lime_gen_time += lime_time
                logging.debug(f"Fold {fold_num}, Seed {self.seed}: LIME values generated in {lime_time:.2f} seconds.")

                if explanation_list:
                    for feature_desc, weight in explanation_list:
                         parts = feature_desc.split(' ')
                         feature_name = parts[0] if len(parts) > 1 and parts[1] in ['<=', '>=', '<', '>'] else feature_desc
                         all_lime_results.append({
                             'seed': self.seed, 'fold': fold_num, 'feature': feature_name,
                             'raw_feature_desc': feature_desc, 'lime_importance': weight, 'time_to_generate' : lime_time
                         })

            except AttributeError as ae:
                 logging.error(f"Fold {fold_num}, Seed {self.seed}: LIME AttributeError (likely needs predict_fn): {ae}. Skipping.", exc_info=False) # Less verbose log
            except Exception as e:
                logging.error(f"Fold {fold_num}, Seed {self.seed}: Error generating LIME values: {e}", exc_info=False)

        logging.info(f"Seed {self.seed}: Finished generating LIME values. Total time: {total_lime_gen_time:.2f}s")
        return pd.DataFrame(all_lime_results) if all_lime_results else pd.DataFrame()


    def _generate_shap_explanations(self):

        shap_explainers = self.results.get('shap_explainers', [])
        samples = self.results.get('samples', [])
        all_shap_results = []
        total_shap_gen_time = 0

        logging.info(f"Seed {self.seed}: Generating SHAP explanation values...")

        for i in range(self.num_folds):
            fold_num = i + 1
            shap_explainer = shap_explainers[i] if i < len(shap_explainers) else None
            fraud_sample = samples[i] if i < len(samples) else None

            if shap_explainer is None or fraud_sample is None or fraud_sample.empty:
                logging.warning(f"Fold {fold_num}, Seed {self.seed}: SHAP explainer or sample missing, skipping explanation.")
                continue

            shap_start_time = time.time()
            shap_values_df = None
            try:
                shap_values = shap_explainer(fraud_sample.values.reshape(1,-1)).values
                if (shap_values.ndim == 3):
                  fraud_shap_values = shap_values[0][:, 1]
                  fraud_shap_values_reset = pd.Series(fraud_shap_values, index=pd.Index(fraud_sample.keys()))
                  fraud_shap_values_abs = np.abs(fraud_shap_values_reset.values)
                  fraud_shap_values_abs = fraud_shap_values_abs.reshape(1, -1)
                  mean_shap_values = fraud_shap_values_abs.mean(axis=0)
                else:
                  mean_shap_values = np.abs(shap_values).mean(axis=0)

                shap_values_df = pd.DataFrame({
                    'feature': fraud_sample.keys(),
                    'shap_importance': mean_shap_values
                })
  
                shap_time = time.time() - shap_start_time
                total_shap_gen_time += shap_time
                logging.debug(f"Fold {fold_num}, Seed {self.seed}: SHAP values generated in {shap_time:.2f} seconds.")

                if shap_values_df is not None and not shap_values_df.empty:
                    shap_values_df['seed'] = self.seed
                    shap_values_df['fold'] = fold_num
                    shap_values_df['time_to_generate'] = shap_time
                    all_shap_results.append(shap_values_df)

            except Exception as e:
                logging.error(f"Fold {fold_num}, Seed {self.seed}: Error generating SHAP values: {e}", exc_info=False)

        logging.info(f"Seed {self.seed}: Finished generating SHAP values. Total time: {total_shap_gen_time:.2f}s")
        return pd.concat(all_shap_results, ignore_index=True) if all_shap_results else pd.DataFrame()

    def save_explanations(self):

        lime_final_df = self._generate_lime_explanations()
        if not lime_final_df.empty:

            try:

                lime_filepath = self.results_dir / "lime_explanations.csv"
                cols_order = ['seed', 'fold', 'feature', 'raw_feature_desc', 'lime_importance', 'time_to_generate']
                lime_final_df = lime_final_df[[col for col in cols_order if col in lime_final_df.columns]]
                lime_final_df.to_csv(lime_filepath, index=False, float_format='%.6f')

                logging.info(f"Seed {self.seed}: Saved LIME explanations to {lime_filepath}")
            except Exception as e:

                 logging.error(f"Seed {self.seed}: Error saving LIME explanations: {e}", exc_info=True)
                 lime_final_df = pd.DataFrame()
        else:
             logging.warning(f"Seed {self.seed}: No LIME explanations generated to save.")


        shap_final_df = self._generate_shap_explanations()
        if not shap_final_df.empty:

             try:
                shap_filepath = self.results_dir / "shap_explanations.csv"
                cols_order = ['seed', 'fold', 'feature', 'shap_importance', 'time_to_generate']
                shap_final_df = shap_final_df[[col for col in cols_order if col in shap_final_df.columns]]
                shap_final_df.to_csv(shap_filepath, index=False, float_format='%.6f')

                logging.info(f"Seed {self.seed}: Saved SHAP explanations to {shap_filepath}")
             except Exception as e:
                logging.error(f"Seed {self.seed}: Error saving SHAP explanations: {e}", exc_info=True)
                shap_final_df = pd.DataFrame()
        else:
             logging.warning(f"Seed {self.seed}: No SHAP explanations generated to save.")

        return lime_final_df, shap_final_df

    def save_ground_truth(self):

        models = self.results.get('models', [])
        importances_list = []
        importance_col_name = 'model_importance' 

        for i, model in enumerate(models):

            fold_num = i + 1
            fold_importances = None

            try:
              coef_ = model.coef_
              abs_coef = np.mean(np.abs(coef_), axis=0)

              fold_importances = pd.DataFrame({'feature': self.feature_names, importance_col_name: abs_coef})

              if len(abs_coef) == len(self.feature_names): 
                logging.info(f"Fold {fold_num}, Seed {self.seed}: Everything is fine with abs coef.")
              else: 
                logging.warning(f"Fold {fold_num}, Seed {self.seed}: Mismatch len(coef_) != len(feature_names)")

              if fold_importances is not None:
                fold_importances['seed'] = self.seed
                fold_importances['fold'] = fold_num
                importances_list.append(fold_importances)

            except Exception as e:
                logging.error(f"Seed {self.seed}, Fold {fold_num}: Error extracting model importance: {e}", exc_info=True)

        if not importances_list:
            logging.warning(f"Seed {self.seed}: No model importances extracted.")
            return pd.DataFrame()

        try:
            combined_df = pd.concat(importances_list, ignore_index=True)
            filepath = self.results_dir / "model_importances.csv"
            cols_order = ['seed', 'fold', 'feature', importance_col_name]
            combined_df = combined_df[[col for col in cols_order if col in combined_df.columns]]
            combined_df.to_csv(filepath, index=False, float_format='%.6f')

            logging.info(f"Seed {self.seed}: Saved model importances to {filepath}")
            return combined_df

        except Exception as e:
            logging.error(f"Seed {self.seed}: Error saving model importances: {e}", exc_info=True)

            return pd.DataFrame()

