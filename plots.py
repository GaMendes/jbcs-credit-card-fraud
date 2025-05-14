import pandas as pd
from pathlib import Path
import logging
import glob
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

default_pos_color = "#008bfb"

class ExplanationPlotter:

    """
    Generates feature importance plots based on saved experiment explanation files.
    Plots show absolute importance values in descending order. Only the first seed (experiment1) is plotted in this work.
    """

    def __init__(self, results_base_dir="results", plots_base_dir="results/plots", top_n_features=10):

        """
        Initializes the ExplanationPlotter.

        Args:
            results_base_dir (str): The base directory where experiment result CSVs are stored.
            plots_base_dir (str): The base directory where generated plot files will be saved.
            top_n_features (int): The number of top features to plot.
        """

        self.results_base_dir = Path(results_base_dir)
        self.plots_base_dir = Path(plots_base_dir)
        self.top_n = top_n_features

        self.lime_col   = 'lime_importance'
        self.shap_col   = 'shap_importance'
        self.logreg_col = 'model_importance'
        self.logreg_model_name = 'LogisticRegression'

        try:
            self.plots_base_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Plots directory created/verified: {self.plots_base_dir}")

        except OSError as e:
            logging.error(f"Failed to create plots directory {self.plots_base_dir}: {e}")
            return

        logging.info(f"ExplanationPlotter initialized. Results base: '{self.results_base_dir}', Plots base: '{self.plots_base_dir}', Top N: {self.top_n}")

    def _find_files(self, file_pattern, base_dir):

        """Helper to find specific files within a given base directory."""

        pattern = str(base_dir / file_pattern)
        files = [Path(f) for f in glob.glob(pattern)]

        logging.debug(f"Search pattern: {pattern}, Found: {len(files)} files.")
        if not files:
             logging.warning(f"No files found for pattern: {pattern}")
        return files

    def _get_top_n_features_for_plot(self, df, importance_col):

        """
        Gets top N features and their ABSOLUTE importance values for plotting,
        sorted DESCENDING by absolute importance.

        Returns:
            pd.DataFrame: DataFrame with 'feature' and 'abs_importance' columns for top N,
                          sorted descending by absolute importance value. Returns empty DF if error.
        """

        if df is None or df.empty or importance_col not in df.columns or 'feature' not in df.columns:
            return pd.DataFrame(columns=['feature', 'abs_importance'])

        df_copy = df.copy()
        df_copy[importance_col] = pd.to_numeric(df_copy[importance_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[importance_col, 'feature'])
        if df_copy.empty:
            return pd.DataFrame(columns=['feature', 'abs_importance'])

        df_copy['abs_importance'] = df_copy[importance_col].abs()
        agg_df = df_copy.groupby('feature')['abs_importance'].mean().reset_index()
        top_n_features_df = agg_df.nlargest(self.top_n, 'abs_importance')
        top_n_sorted_df = top_n_features_df.sort_values('abs_importance', ascending=False)

        return top_n_sorted_df[['feature', 'abs_importance']]

    def _plot_feature_importance_subplots(self, data_dict, plot_title, filename_base, importance_col, y_label="Feature Name"):

        """
        Internal helper to create 5-subplot feature importance plots using ABSOLUTE values,
        ordered descendingly.

        Args:
            data_dict (dict): {fold_number: pd.DataFrame_with_feature_and_importance_cols}
            plot_title (str): The main title for the figure.
            filename_base (Path): The base path and filename stem for saving the plot
                                  (e.g., results/plots/dataset/model/model_lime).
            importance_col (str): The name of the column containing original importance values.
            y_label (str): Label for the Y-axis.
        """

        num_folds = 5
        fig, axes = plt.subplots(nrows=1, ncols=num_folds, figsize=(18, max(4, self.top_n * 0.4)), sharey=False)

        valid_folds = sorted([f for f in data_dict.keys() if not data_dict[f].empty])
        if not valid_folds:
            logging.warning(f"No data found for any fold for plot: {plot_title}. Skipping plot generation.")
            plt.close(fig)
            return

        if len(valid_folds) < num_folds:
             for i in range(len(valid_folds), num_folds): axes[i].set_visible(False)

        fig.tight_layout(pad=3.5, rect=[0, 0.03, 1, 0.95])

        max_abs_importance = 0
        for fold_num in valid_folds:
             fold_df = data_dict[fold_num]
             plot_data = self._get_top_n_features_for_plot(fold_df, importance_col)
             if not plot_data.empty:
                current_max = plot_data['abs_importance'].max()
                if not np.isnan(current_max) and current_max > max_abs_importance:
                    max_abs_importance = current_max

        for i, fold_num in enumerate(valid_folds):
            ax = axes[i]
            fold_df = data_dict[fold_num]
            plot_data = self._get_top_n_features_for_plot(fold_df, importance_col)

            if plot_data.empty:
                logging.warning(f"No features to plot for Fold {fold_num} in {plot_title}")

                ax.set_yticks([])
                ax.set_xticks([])
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                continue

            plot_data_reversed = plot_data.iloc[::-1]
            bars = ax.barh(plot_data_reversed['feature'], plot_data_reversed['abs_importance'], align='center', color=default_pos_color)
            ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)

            for bar in bars:
                val = bar.get_width()
                offset = max_abs_importance * 0.01 if max_abs_importance > 0 else 0.01
                ax.text(val + offset, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                        ha='left', va='center', color=default_pos_color, fontsize=8)

            if i == 0: ax.set_ylabel(y_label)

        plt.suptitle(plot_title, fontsize=14)

        plot_output_dir = filename_base.parent
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = f"{filename_base.name}.pdf"
        plot_filepath = plot_output_dir / plot_filename

        try:
            plt.savefig(plot_filepath, format="pdf", bbox_inches='tight')
            logging.info(f"Saved plot: {plot_filepath}")
        except Exception as e: logging.error(f"Error saving plot {plot_filepath}: {e}", exc_info=True)
        plt.close(fig)

    def _get_model_name_formatted(self, model_name):

        """
        Returns a formatted string for the dataset name
        Args:
            model_name: unformatted model name string
        """

        if (model_name == 'MLP'): return 'Neural Network'
        if (model_name == 'RandomForest'): return 'Random Forest'
        if (model_name == 'LogisticRegression'): return 'Logistic Regression'
        if (model_name == 'LinearSVC'): return 'SVM'
        if (model_name == 'DecisionTree'): return 'Decision Tree'

        return model_name
    
    def _get_dataset_name_formatted(self, dataset_name):

        """
        Returns a formatted string for the dataset name
        Args:
            dataset_name: unformatted dataset name string
        """

        if (dataset_name == 'synthetic'): return 'Credit card transaction'
        if (dataset_name == 'kaggle'):    return 'Kaggle fraud detection'
        if (dataset_name == 'ieee-cis'): return 'IEEE-CIS fraud detection'

        return dataset_name
    
    def _get_old_file_name(self, model_name, explanability_type=''):

        """
        Returns old filenames for different models.

        Args:
            model_name: unformatted model name string
            explanability_type: shap or lime
        """

        if (model_name == 'MLP' and explanability_type == 'shap'): return 'ann_feature_importance_plot(shap)'
        if (model_name == 'MLP' and explanability_type == 'lime'): return 'ann_feature_importance_plot(lime)'

        if (model_name == 'RandomForest' and explanability_type == 'shap'): return 'random_forest_feature_importance_plot(shap)'
        if (model_name == 'RandomForest' and explanability_type == 'lime'): return 'random_forest_feature_importance_plot(lime)'

        if (model_name == 'LinearSVC' and explanability_type == 'shap'): return 'svm_feature_importance_plot(shap)'
        if (model_name == 'LinearSVC' and explanability_type == 'lime'): return 'svm_feature_importance_plot(lime)'

        if (model_name == 'DecisionTree' and explanability_type == 'shap'): return 'decision_tree_feature_importance_plot(shap)'
        if (model_name == 'DecisionTree' and explanability_type == 'lime'): return 'decision_tree_feature_importance_plot(lime)'

        if (model_name == 'LogisticRegression' and explanability_type == ''):     return 'logistic_regression_feature_importance_plot'
        if (model_name == 'LogisticRegression' and explanability_type == 'shap'): return 'logistic_regression_feature_importance_plot(shap)'
        if (model_name == 'LogisticRegression' and explanability_type == 'lime'): return 'logistic_regression_feature_importance_plot(lime)'

        return model_name

    def generate_plots_for_experiment1(self):

        """
        Generates feature importance plots for experiment1 based on saved CSV files.
        Saves plots to the dedicated plots directory (results/plots/).
        """

        logging.info("Starting generation of explanation plots for experiment1...")
        exp1_dir = self.results_base_dir / "experiment1"

        if not exp1_dir.is_dir():
            logging.warning(f"Experiment 1 directory not found: {exp1_dir}. Skipping plot generation.")
            return

        dataset_dirs = [d for d in exp1_dir.iterdir() if d.is_dir()]
        if not dataset_dirs: logging.warning(f"No dataset directories found in {exp1_dir}."); return

        for ds_dir in dataset_dirs:
            dataset_name = ds_dir.name
            logging.info(f"--- Generating plots for Dataset: {dataset_name} (Experiment 1) ---")

            model_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
            if not model_dirs: logging.warning(f"No model directories found in {ds_dir}."); continue

            for model_dir in model_dirs:
                model_name = model_dir.name
                logging.info(f"--- Processing Model: {model_name} ---")

                plot_output_base = self.plots_base_dir / dataset_name / model_name

                if model_name == self.logreg_model_name:
                    logreg_imp_file = model_dir / "model_importances.csv"
                    if logreg_imp_file.is_file():
                        try:
                            logreg_df = pd.read_csv(logreg_imp_file)
                            if 'fold' in logreg_df.columns:
                                 logreg_data_by_fold = {fold: df for fold, df in logreg_df.groupby('fold')}
                                 self._plot_feature_importance_subplots(
                                     data_dict=logreg_data_by_fold,
                                     plot_title=self._get_dataset_name_formatted(dataset_name),
                                     filename_base=plot_output_base.parent / self._get_old_file_name(model_name),
                                     importance_col=self.logreg_col,
                                     y_label=""
                                 )
                            else: 
                                logging.warning(f"'fold' column missing in LogReg importance file: {logreg_imp_file}")
                        except pd.errors.EmptyDataError: 
                            logging.warning(f"LogReg importance file is empty: {logreg_imp_file}")
                        except Exception as e: 
                            logging.error(f"Error plotting LogReg coefficients for {dataset_name}: {e}", exc_info=True)
                    else:
                        logging.warning(f"LogReg importance file not found: {logreg_imp_file}")

                lime_file = model_dir / "lime_explanations.csv"
                if lime_file.is_file():
                    try:
                        lime_df = pd.read_csv(lime_file)
                        if 'fold' in lime_df.columns:
                             lime_data_by_fold = {fold: df for fold, df in lime_df.groupby('fold')}
                             self._plot_feature_importance_subplots(
                                 data_dict=lime_data_by_fold,
                                 plot_title=self._get_model_name_formatted(model_name),
                                 filename_base=plot_output_base.parent / self._get_old_file_name(model_name, 'lime'),
                                 importance_col=self.lime_col,
                                 y_label=""
                             )
                        else: 
                            logging.warning(f"'fold' column missing in LIME file: {lime_file}")
                    except pd.errors.EmptyDataError: 
                        logging.warning(f"LIME explanations file is empty: {lime_file}")
                    except Exception as e:
                        logging.error(f"Error plotting LIME for {model_name}/{dataset_name}: {e}", exc_info=True)
                else:
                    logging.warning(f"LIME explanations file not found: {lime_file}")

                shap_file = model_dir / "shap_explanations.csv"
                if shap_file.is_file():
                    try:
                        shap_df = pd.read_csv(shap_file)
                        if 'fold' in shap_df.columns:
                             shap_data_by_fold = {fold: df for fold, df in shap_df.groupby('fold')}
                             self._plot_feature_importance_subplots(
                                 data_dict=shap_data_by_fold,
                                 plot_title=self._get_model_name_formatted(model_name),
                                 filename_base=plot_output_base.parent / self._get_old_file_name(model_name, 'shap'),
                                 importance_col=self.shap_col,
                                 y_label=""
                             )
                        else: 
                            logging.warning(f"'fold' column missing in SHAP file: {shap_file}")
                    except pd.errors.EmptyDataError:
                        logging.warning(f"SHAP explanations file is empty: {shap_file}")
                    except Exception as e:
                        logging.error(f"Error plotting SHAP for {model_name}/{dataset_name}: {e}", exc_info=True)
                else:
                    logging.warning(f"SHAP explanations file not found: {shap_file}")

        logging.info("Finished generation of explanation plots for experiment1.")


if __name__ == "__main__":
    logging.info("Executing plotting script...")

    plotter = ExplanationPlotter(results_base_dir="results", plots_base_dir="results/plots", top_n_features=10)
    plotter.generate_plots_for_experiment1()

    logging.info("Plotting script finished.")

