import pandas as pd
from pathlib import Path
import logging
import glob
import numpy as np
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetricsAnalyzer:

    """
    Analyzes metrics, explanation agreement (between LIME, SHAP and ground thruth), and timings from experiment results.
    Includes functionality to fix LIME feature names.
    """

    def __init__(self, results_base_dir="results", top_n_features=10):

        """
        Initializes the MetricsAnalyzer.

        Args:
            results_base_dir (str): The base directory where experiment results are stored
                                    and where summary files will be saved.
            top_n_features (int): The number of top features to consider for agreement.
        """

        self.results_base_dir = Path(results_base_dir)
        self.top_n = top_n_features
        self.metric_columns = [
            'balanced_accuracy', 'sensitivity_recall', 'specificity',
            'precision'
        ]

        self.lime_col          = 'lime_importance'
        self.shap_col          = 'shap_importance'
        self.logreg_col        = 'model_importance'
        self.ground_truth      = 'LogisticRegression'

        # Calculate total possible agreements (5 folds * 5 experiments * top_n features)
        number_experiments          = 5 # Assuming 5 experiments (seeds 1-5)
        number_folds_per_experiment = 5
        self.total_possible_agreements = number_experiments * number_folds_per_experiment * self.top_n

        self.overall_metrics_summary_df  = None
        self.overall_agreement_counts_df = None
        self.overall_agreement_percentage_df = None
        self.timing_summaries_by_dataset = {}

        logging.info(f"MetricsAnalyzer initialized. Results/Summary base: '{self.results_base_dir}', Top N: {self.top_n}, Total Possible Agreements: {self.total_possible_agreements}")

    def _find_files(self, file_pattern="metrics.csv", base_dir=None):

        """
        Helper to find specific files within a given base directory structure.
        Handles patterns like 'metrics.csv' or 'ModelName/filename.csv'.
        Also handles searching for files directly in dataset dirs like 'agreement_counts.csv'.
        """

        search_base = base_dir if base_dir else self.results_base_dir

        if file_pattern == "agreement_counts.csv":
            pattern = str(search_base / "experiment*" / "*" / file_pattern)

        elif file_pattern == "lime_explanations.csv":
            pattern = str(search_base / "experiment*" / "*" / "*" / file_pattern)

        elif file_pattern == "shap_explanations.csv":
            pattern = str(search_base / "experiment*" / "*" / "*" / file_pattern)

        elif file_pattern == "timings.csv":
            pattern = str(search_base / "experiment*" / "*" / "*" / file_pattern)

        elif "/" in file_pattern:
            model_name, filename = file_pattern.split('/', 1)
            pattern = str(self.results_base_dir / "experiment*" / "*" / model_name / filename)

        else:
             if base_dir:
                pattern = str(search_base / "*" / file_pattern)
             else:
                pattern = str(search_base / "experiment*" / "*" / "*" / file_pattern)

        files = [Path(f) for f in glob.glob(pattern)]
        logging.info(f"Found {len(files)} files matching pattern: {pattern} (for input: '{file_pattern}')")

        if not files:
            logging.warning(f"No files found for pattern: {pattern}")

        return files

    def _get_top_n_features(self, df, importance_col):

        """
        Gets top N features based on absolute importance from a DataFrame.

        Returns:
            tuple: (set_of_feature_names, list_of_tuples_sorted_by_abs_importance)
                   Returns (set(), []) if no features found.
                   List format: [(feature_name, abs_importance), ...] sorted ascending by abs_importance.
        """

        if df is None or df.empty or importance_col not in df.columns or 'feature' not in df.columns:
            return set(), []

        df_copy = df.copy()
        df_copy[importance_col] = pd.to_numeric(df_copy[importance_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[importance_col, 'feature'])

        if df_copy.empty:
            return set(), []

        df_copy['abs_importance'] = df_copy[importance_col].abs()
        agg_df = df_copy.groupby('feature')['abs_importance'].mean().reset_index()

        top_n_df = agg_df.nlargest(self.top_n, 'abs_importance')
        top_n_sorted_list = sorted(
            [(row['feature'], row['abs_importance']) for _, row in top_n_df.iterrows()],
            key=lambda item: item[1]
        )
        top_feature_names_set = set(top_n_df['feature'].unique())

        return top_feature_names_set, top_n_sorted_list

    def fix_lime_feature_names(self):

        """
        Finds all lime_explanations.csv files and corrects the 'feature' column
        by extracting the name from 'raw_feature_desc'. Overwrites the files.
        """

        logging.info("Starting process to fix LIME feature names...")

        lime_files = self._find_files("lime_explanations.csv")
        if not lime_files: 
            logging.warning("No 'lime_explanations.csv' files found to fix.")
            return

        feature_regex = re.compile(r"\b([a-zA-Z_]\w*)\b")

        files_processed = 0
        files_failed = 0

        for file_path in lime_files:

            logging.debug(f"Processing LIME file: {file_path}")

            try:

                df = pd.read_csv(file_path)
                if 'raw_feature_desc' not in df.columns or 'feature' not in df.columns:
                    logging.warning(f"Skipping file {file_path}: Missing 'raw_feature_desc' or 'feature' column."); files_failed += 1; continue

                def extract_name(desc):
                    desc_str = str(desc); potential_features = feature_regex.findall(desc_str)
                    if '<' in desc_str or '>' in desc_str:
                        for feature in potential_features:
                            try: 
                                float(feature)
                            except ValueError: 
                                return feature
                        logging.warning(f"Could not isolate feature name from numerical desc: '{desc_str}'. Using original."); return desc_str
                    else: 
                        return potential_features[0] if potential_features else desc_str
    
                df['feature'] = df['raw_feature_desc'].apply(extract_name)
                df.to_csv(file_path, index=False, float_format='%.6f')

                logging.info(f"Corrected and saved: {file_path}")
                files_processed += 1

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty LIME file: {file_path}")
                files_failed += 1
            except Exception as e: 
                logging.error(f"Error processing LIME file {file_path}: {e}", exc_info=True)
                files_failed += 1
        logging.info(f"Finished fixing LIME feature names. Processed: {files_processed}, Failed/Skipped: {files_failed}")

    def calculate_mean_metrics_per_run(self):

        """
        Calculates the mean of metrics for each specific experiment run
        and saves summaries per model run and per dataset run.
        """

        logging.info("Starting calculation of mean metrics per run and per dataset summary...")

        metric_files = self._find_files("metrics.csv")
        if not metric_files: 
            return

        dataset_experiment_summaries = {}

        for metric_file_path in metric_files:
            try:
                model_name   = metric_file_path.parent.name
                dataset_name = metric_file_path.parent.parent.name
                experiment_run_name = metric_file_path.parent.parent.parent.name

                logging.info(f"Processing metrics: {metric_file_path}")

                df = pd.read_csv(metric_file_path)
                for col in self.metric_columns:
                    if col not in df.columns: df[col] = np.nan

                mean_metrics_for_model = df[self.metric_columns].mean().to_frame().T
                mean_metrics_for_model.insert(0, 'algorithm', model_name)
                summary_per_model_run_path = metric_file_path.parent / "mean_metrics_summary_per_run.csv"

                mean_metrics_for_model.to_csv(summary_per_model_run_path, index=False, float_format='%.6f')

                logging.info(f"Saved model run summary: {summary_per_model_run_path}")

                summary_key = (experiment_run_name, dataset_name)
                dataset_experiment_summaries.setdefault(summary_key, []).append(mean_metrics_for_model)

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty metrics file: {metric_file_path}")

            except Exception as e: 
                logging.error(f"Error processing metrics file {metric_file_path}: {e}", exc_info=True)

        logging.info("Generating dataset-level mean metrics summaries...")

        for (exp_run, ds_name), model_means_list in dataset_experiment_summaries.items():

            if not model_means_list: 
                continue

            try:
                combined_dataset_summary_df = pd.concat(model_means_list, ignore_index=True)
                cols_order = ['algorithm'] + [col for col in self.metric_columns if col in combined_dataset_summary_df.columns]
                combined_dataset_summary_df = combined_dataset_summary_df[cols_order]

                dataset_summary_file_path = self.results_base_dir / exp_run / ds_name / "all_algorithms_dataset_mean_metrics.csv"
                combined_dataset_summary_df.to_csv(dataset_summary_file_path, index=False, float_format='%.6f')

                logging.info(f"Saved dataset summary: {dataset_summary_file_path}")
            except Exception as e: 
                logging.error(f"Error creating dataset summary for {exp_run}/{ds_name}: {e}", exc_info=True)

        logging.info("Finished calculation of mean metrics per run and per dataset summary.")

    def calculate_overall_mean_metrics(self):

        """
        Calculates the overall mean and standard deviation of metrics for each
        algorithm/dataset across all runs.
        Saves the summary to 'overall_metrics_summary_with_std.csv' in the results directory.
        Stores the DataFrame in self.overall_metrics_summary_df.
        """

        logging.info("Starting calculation of overall metric statistics (mean and std dev)...")

        metric_files = self._find_files("metrics.csv")
        if not metric_files:
            self.overall_metrics_summary_df = pd.DataFrame()
            return

        all_metrics_data = []

        for metric_file_path in metric_files:
            try:
                model_name   = metric_file_path.parent.name
                dataset_name = metric_file_path.parent.parent.name

                df = pd.read_csv(metric_file_path)
                df['dataset']   = dataset_name
                df['algorithm'] = model_name
    
                all_metrics_data.append(df)

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty metrics file: {metric_file_path}")
            except Exception as e: 
                logging.error(f"Error reading metrics file {metric_file_path}: {e}", exc_info=True)

        if not all_metrics_data:
            self.overall_metrics_summary_df = pd.DataFrame()

            logging.warning("No data collected from metric files. Cannot generate overall stats summary.")
            return

        combined_df = pd.concat(all_metrics_data, ignore_index=True)

        for col in self.metric_columns:
            if col not in combined_df.columns:
                logging.warning(f"Overall: Metric column '{col}' not found in combined data. Will be NaN.")
                combined_df[col] = np.nan

            else:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        # Group by dataset and algorithm, then calculate mean and std for the metric columns
        # The stats are calculated across all folds from all relevant experiment runs
        overall_stats_df = combined_df.groupby(['dataset', 'algorithm'])[self.metric_columns].agg(['mean', 'std'])

        overall_stats_df.columns = ['_'.join(col).strip() for col in overall_stats_df.columns.values]
        overall_stats_df = overall_stats_df.reset_index()

        self.overall_metrics_summary_df = overall_stats_df
        summary_file_path = self.results_base_dir / "overall_metrics_summary_with_std.csv"

        try:
            self.overall_metrics_summary_df.to_csv(summary_file_path, index=False, float_format='%.6f')

            logging.info(f"Saved overall metrics summary (with std dev) to: {summary_file_path}")

        except Exception as e: 
            logging.error(f"Error saving overall metrics summary (with std dev): {e}", exc_info=True)
        logging.info("Finished calculation of overall metric statistics.")

    def calculate_and_save_per_experiment_agreement(self):

        """
        Calculates agreement counts and top features per experiment run and dataset.
        Saves 'agreement_counts.csv' and 'top_features_for_validation.csv'
        in the 'results/experiment<N>/<dataset_name>/' directory.
        """

        logging.info(f"Starting per-experiment agreement calculation (Top {self.top_n} vs LogReg)...")

        experiment_dirs = [d for d in self.results_base_dir.iterdir() if d.is_dir() and d.name.startswith("experiment")]
        if not experiment_dirs: 
            logging.warning("No experiment directories found.")
            return

        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name

            logging.info(f"--- Processing Experiment: {exp_name} ---")

            dataset_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
            if not dataset_dirs: 
                logging.warning(f"No dataset directories found in {exp_name}.")
                continue

            for ds_dir in dataset_dirs:
                dataset_name = ds_dir.name

                logging.info(f"--- Processing Dataset: {dataset_name} in {exp_name} ---")

                logreg_top_features_per_fold = {}

                logreg_imp_file = ds_dir / self.ground_truth / "model_importances.csv"
                if not logreg_imp_file.is_file():
                    logging.warning(f"Ground truth file not found: {logreg_imp_file}. Skipping agreement for {exp_name}/{dataset_name}.")
                    continue

                try:
                    logreg_df = pd.read_csv(logreg_imp_file)

                    for fold in logreg_df['fold'].unique():

                        fold_df = logreg_df[logreg_df['fold'] == fold]
                        top_set, _ = self._get_top_n_features(fold_df, self.logreg_col)

                        if top_set: 
                            logreg_top_features_per_fold[fold] = top_set
                        else: 
                            logging.warning(f"Could not extract LogReg top features for Fold {fold} in {exp_name}/{dataset_name}")

                except pd.errors.EmptyDataError: 
                    logging.warning(f"LogReg importance file is empty: {logreg_imp_file}. Skipping agreement.")
                    continue
                except Exception as e: 
                    logging.error(f"Error processing LogReg importance file {logreg_imp_file}: {e}", exc_info=True); 
                    continue
                if not logreg_top_features_per_fold: 
                    logging.warning(f"No ground truth features loaded for {exp_name}/{dataset_name}. Skipping agreement.")
                    continue

                model_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
                agreement_counts_data = []
                top_features_data     = []

                for model_dir in model_dirs:
                    model_name = model_dir.name
                    logging.debug(f"Processing model {model_name} for agreement in {exp_name}/{dataset_name}")

                    lime_file = model_dir / "lime_explanations.csv"
                    lime_df = pd.DataFrame()
                    if lime_file.is_file():
                        try: lime_df = pd.read_csv(lime_file)
                        except Exception as e: 
                            logging.error(f"Error reading LIME file {lime_file}: {e}")

                    shap_file = model_dir / "shap_explanations.csv"
                    shap_df = pd.DataFrame()
                    if shap_file.is_file():
                        try: shap_df = pd.read_csv(shap_file)
                        except Exception as e: 
                            logging.error(f"Error reading SHAP file {shap_file}: {e}")

                    folds_present = set()
                    if not lime_df.empty: 
                        folds_present.update(lime_df['fold'].unique())
                    if not shap_df.empty: 
                        folds_present.update(shap_df['fold'].unique())

                    for fold in sorted(list(folds_present)):
                        ground_truth_set = logreg_top_features_per_fold.get(fold)
                        if ground_truth_set is None: 
                            logging.warning(f"Ground truth missing for fold {fold} in {exp_name}/{dataset_name}. Skipping comparisons.")
                            continue

                        lime_fold_df = lime_df[lime_df['fold'] == fold] if not lime_df.empty else pd.DataFrame()
                        top_lime_set, top_lime_sorted_list = self._get_top_n_features(lime_fold_df, self.lime_col)
                        lime_intersect_count = len(top_lime_set.intersection(ground_truth_set)) if top_lime_set else 0
                        lime_sorted_feature_names = [f for f, imp in top_lime_sorted_list]

                        shap_fold_df = shap_df[shap_df['fold'] == fold] if not shap_df.empty else pd.DataFrame()
                        top_shap_set, top_shap_sorted_list = self._get_top_n_features(shap_fold_df, self.shap_col)
                        shap_intersect_count = len(top_shap_set.intersection(ground_truth_set)) if top_shap_set else 0
                        shap_sorted_feature_names = [f for f, imp in top_shap_sorted_list]

                        agreement_counts_data.append({'algorithm': model_name, 'fold': fold, 'lime_logreg_intersection_count': lime_intersect_count, 'shap_logreg_intersection_count': shap_intersect_count})
                        top_features_data.append({'algorithm': model_name, 'fold': fold, 'method': 'LIME', 'top_features_ordered': ', '.join(lime_sorted_feature_names)})
                        top_features_data.append({'algorithm': model_name, 'fold': fold, 'method': 'SHAP', 'top_features_ordered': ', '.join(shap_sorted_feature_names)})

                if agreement_counts_data:
                    try:
                        agreement_df = pd.DataFrame(agreement_counts_data).sort_values(by=['algorithm', 'fold'])
                        agreement_file_path = ds_dir / "agreement_counts.csv"
                        agreement_df.to_csv(agreement_file_path, index=False)

                        logging.info(f"Saved agreement counts to: {agreement_file_path}")

                    except Exception as e: 
                        logging.error(f"Error saving agreement counts for {exp_name}/{dataset_name}: {e}", exc_info=True)
                else: 
                    logging.warning(f"No agreement counts generated for {exp_name}/{dataset_name}.")

                if top_features_data:
                    try:
                        top_features_df = pd.DataFrame(top_features_data).sort_values(by=['algorithm', 'fold', 'method'])
                        top_features_df = top_features_df.rename(columns={'top_features_ordered': 'top_10_features_ordered_by_abs_importance'})
                        top_features_file_path = ds_dir / "top_features_for_validation.csv"
                        top_features_df.to_csv(top_features_file_path, index=False)

                        logging.info(f"Saved top features for validation to: {top_features_file_path}")
                    except Exception as e: 
                        logging.error(f"Error saving top features for {exp_name}/{dataset_name}: {e}", exc_info=True)
                else: 
                    logging.warning(f"No top features data generated for {exp_name}/{dataset_name}.")

        logging.info("Finished per-experiment agreement calculation.")

    def calculate_and_save_overall_agreement_tables(self):

        """
        Calculates the overall agreement SUM and PERCENTAGE for each algorithm/dataset
        based on the 'agreement_counts.csv' files. Saves tables to the results directory.
        Stores DataFrames in self.overall_agreement_counts_df and self.overall_agreement_percentage_df.
        """

        logging.info("Starting calculation of overall agreement tables (Counts and Percentages)...")

        agreement_files = self._find_files("agreement_counts.csv")
        if not agreement_files:
            logging.warning("No 'agreement_counts.csv' files found. Cannot calculate overall agreement.")
            return

        all_agreement_data = []

        for file_path in agreement_files:
            try:
                dataset_name = file_path.parent.name
                df = pd.read_csv(file_path)

                df['dataset'] = dataset_name
                all_agreement_data.append(df)

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty agreement counts file: {file_path}")
            except Exception as e: 
                logging.error(f"Error reading agreement counts file {file_path}: {e}", exc_info=True)

        if not all_agreement_data:
            logging.warning("No data loaded from agreement count files. Cannot generate overall agreement summary.")
            return

        combined_df = pd.concat(all_agreement_data, ignore_index=True)
        agg_agreement = combined_df.groupby(['dataset', 'algorithm'])[[
            'lime_logreg_intersection_count',
            'shap_logreg_intersection_count'
        ]].sum().reset_index()

        agg_agreement_counts = agg_agreement.rename(columns={
            'lime_logreg_intersection_count': 'LIME Total Agreements',
            'shap_logreg_intersection_count': 'SHAP Total Agreements'
        })


        self.overall_agreement_percentage_df = None
        if self.total_possible_agreements > 0:
            agg_agreement['LIME Agreement (%)'] = (agg_agreement['lime_logreg_intersection_count'] / self.total_possible_agreements) * 100
            agg_agreement['SHAP Agreement (%)'] = (agg_agreement['shap_logreg_intersection_count'] / self.total_possible_agreements) * 100
        else:
             logging.error("Total possible agreements is zero or negative. Cannot calculate percentage.")

        try:
            pivot_counts_df = agg_agreement_counts[['dataset', 'algorithm', 'LIME Total Agreements', 'SHAP Total Agreements']]
            self.overall_agreement_counts_df = pivot_counts_df.pivot_table(
                index='algorithm', columns='dataset',
                values=['LIME Total Agreements', 'SHAP Total Agreements']
            ).swaplevel(0, 1, axis=1).sort_index(axis=1, level=[0, 1])

            counts_summary_file_path = self.results_base_dir / "overall_agreement_total_counts_table.csv" 
            self.overall_agreement_counts_df.to_csv(counts_summary_file_path, float_format='%.0f')

            logging.info(f"Saved overall agreement total counts table to: {counts_summary_file_path}")

        except Exception as e:
             logging.error(f"Error pivoting/formatting/saving agreement counts table: {e}", exc_info=True)
             self.overall_agreement_counts_df = None

        if self.total_possible_agreements > 0:
            try:
                pivot_perc_df = agg_agreement[['dataset', 'algorithm', 'LIME Agreement (%)', 'SHAP Agreement (%)']]
                self.overall_agreement_percentage_df = pivot_perc_df.pivot_table(
                    index='algorithm', columns='dataset',
                    values=['LIME Agreement (%)', 'SHAP Agreement (%)']
                ).swaplevel(0, 1, axis=1).sort_index(axis=1, level=[0, 1])

                perc_summary_file_path = self.results_base_dir / "overall_agreement_total_percentage_table.csv"
                self.overall_agreement_percentage_df.to_csv(perc_summary_file_path, float_format='%.2f')

                logging.info(f"Saved overall agreement total percentage table to: {perc_summary_file_path}")

            except Exception as e:
                 logging.error(f"Error pivoting/formatting/saving agreement percentage table: {e}", exc_info=True)
                 self.overall_agreement_percentage_df = None
        else:
            logging.warning("Skipping saving of percentage table due to invalid total_possible_agreements.")

        logging.info("Finished overall agreement table calculations.")

    def calculate_and_save_timing_summary_tables(self):

        """
        Calculates the mean timing for explainer creation and explanation generation
        for each algorithm/dataset across all experiments.

        Saves separate summary tables per dataset to the results directory.
        """

        logging.info("Starting calculation of timing summary tables...")

        timing_files           = self._find_files("timings.csv")
        lime_explanation_files = self._find_files("lime_explanations.csv")
        shap_explanation_files = self._find_files("shap_explanations.csv")

        if not timing_files:
            logging.warning("No 'timings.csv' files found. Cannot calculate explainer creation times.")

        if not lime_explanation_files and not shap_explanation_files:
            logging.warning("No LIME or SHAP explanation files found. Cannot calculate explanation generation times.")
            return 

        all_creation_times = []
        for file_path in timing_files:
            try:
                model_name      = file_path.parent.name
                dataset_name    = file_path.parent.parent.name
                experiment_name = file_path.parent.parent.parent.name
                seed = int(experiment_name.replace("experiment", ""))

                df = pd.read_csv(file_path)
                df['dataset']   = dataset_name
                df['algorithm'] = model_name
                df['seed']      = seed
                all_creation_times.append(df)

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty timings file: {file_path}")
            except ValueError: 
                logging.warning(f"Could not parse seed from path for timings: {file_path}")
            except Exception as e: 
                logging.error(f"Error reading timings file {file_path}: {e}", exc_info=True)

        creation_times_df = pd.DataFrame()
        if all_creation_times:
            creation_times_df = pd.concat(all_creation_times, ignore_index=True)

            creation_cols_map = {
                'lime_explainer_creation_time': 'LIME Explainer Time (s)',
                'shap_explainer_creation_time': 'SHAP Explainer Time (s)'
            }

            existing_creation_cols = [col for col in creation_cols_map.keys() if col in creation_times_df.columns]
            if not existing_creation_cols:
                 logging.warning("Expected explainer creation time columns not found in timings.csv.")
                 creation_times_df = pd.DataFrame(columns=['dataset', 'algorithm', 'seed'])
            else:
                 creation_times_df = creation_times_df[['dataset', 'algorithm', 'seed'] + existing_creation_cols]
                 creation_times_df = creation_times_df.rename(columns=creation_cols_map)

        else:
            logging.warning("No explainer creation time data loaded from timings.csv.")
            creation_times_df = pd.DataFrame(columns=['dataset', 'algorithm', 'seed'])

        all_lime_gen_times = []
        for file_path in lime_explanation_files:
            try:
                model_name      = file_path.parent.name
                dataset_name    = file_path.parent.parent.name
                experiment_name = file_path.parent.parent.parent.name
                seed = int(experiment_name.replace("experiment", ""))

                df = pd.read_csv(file_path)
                if 'time_to_generate' in df.columns and 'fold' in df.columns:
  
                    fold_times = df.groupby('fold')['time_to_generate'].first().reset_index()
                    fold_times.rename(columns={'time_to_generate': 'LIME Values Time (s)'}, inplace=True)

                    fold_times['dataset']   = dataset_name
                    fold_times['algorithm'] = model_name
                    fold_times['seed']      = seed

                    all_lime_gen_times.append(fold_times)
                else:
                    logging.warning(f"'time_to_generate' or 'fold' column missing in LIME file: {file_path}")

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty LIME explanations file: {file_path}")
            except ValueError: 
                logging.warning(f"Could not parse seed from path for LIME explanations: {file_path}")
            except Exception as e: 
                logging.error(f"Error reading LIME explanations file {file_path}: {e}", exc_info=True)

        lime_gen_times_df = pd.DataFrame()
        if all_lime_gen_times:
            lime_gen_times_df = pd.concat(all_lime_gen_times, ignore_index=True)
        else:
            logging.warning("No LIME explanation generation time data loaded.")
            lime_gen_times_df = pd.DataFrame(columns=['dataset', 'algorithm', 'seed', 'fold', 'LIME Values Time (s)'])

        all_shap_gen_times = []
        for file_path in shap_explanation_files:
            try:
                model_name      = file_path.parent.name
                dataset_name    = file_path.parent.parent.name
                experiment_name = file_path.parent.parent.parent.name
                seed = int(experiment_name.replace("experiment", ""))

                df = pd.read_csv(file_path)
                if 'time_to_generate' in df.columns and 'fold' in df.columns:
                    fold_times = df.groupby('fold')['time_to_generate'].first().reset_index()
                    fold_times.rename(columns={'time_to_generate': 'SHAP Values Time (s)'}, inplace=True)

                    fold_times['dataset']   = dataset_name
                    fold_times['algorithm'] = model_name
                    fold_times['seed']      = seed

                    all_shap_gen_times.append(fold_times)
                else:
                    logging.warning(f"'time_to_generate' or 'fold' column missing in SHAP file: {file_path}")

            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping empty SHAP explanations file: {file_path}")
            except ValueError: 
                logging.warning(f"Could not parse seed from path for SHAP explanations: {file_path}")
            except Exception as e: 
                logging.error(f"Error reading SHAP explanations file {file_path}: {e}", exc_info=True)

        shap_gen_times_df = pd.DataFrame()
        if all_shap_gen_times:
            shap_gen_times_df = pd.concat(all_shap_gen_times, ignore_index=True)
        else:
            logging.warning("No SHAP explanation generation time data loaded.")
            shap_gen_times_df = pd.DataFrame(columns=['dataset', 'algorithm', 'seed', 'SHAP Values Time (s)'])

        if creation_times_df.empty and lime_gen_times_df.empty and shap_gen_times_df.empty:
             logging.warning("No timing data available from any source. Cannot generate timing summary.")
             return

        all_dfs_for_keys = []
        if not creation_times_df.empty and all(col in creation_times_df.columns for col in ['dataset', 'algorithm', 'seed']):
            all_dfs_for_keys.append(creation_times_df[['dataset', 'algorithm', 'seed']])
        if not lime_gen_times_df.empty and all(col in lime_gen_times_df.columns for col in ['dataset', 'algorithm', 'seed']):
            all_dfs_for_keys.append(lime_gen_times_df[['dataset', 'algorithm', 'seed']])
        if not shap_gen_times_df.empty and all(col in shap_gen_times_df.columns for col in ['dataset', 'algorithm', 'seed']):
            all_dfs_for_keys.append(shap_gen_times_df[['dataset', 'algorithm', 'seed']])

        if not all_dfs_for_keys:
            logging.warning("Could not establish a base for merging timing data. Skipping summary.")
            return

        base_keys_df = pd.concat(all_dfs_for_keys).drop_duplicates().reset_index(drop=True)

        if not creation_times_df.empty and all(col in creation_times_df.columns for col in ['dataset', 'algorithm', 'seed']):
            merged_df = pd.merge(base_keys_df, creation_times_df, on=['dataset', 'algorithm', 'seed'], how='left')
        else:
            merged_df = base_keys_df.copy()
            if 'LIME Explainer Time (s)' not in merged_df.columns: 
                merged_df['LIME Explainer Time (s)'] = np.nan
            if 'SHAP Explainer Time (s)' not in merged_df.columns: 
                merged_df['SHAP Explainer Time (s)'] = np.nan

        if not lime_gen_times_df.empty and all(col in lime_gen_times_df.columns for col in ['dataset', 'algorithm', 'seed']):
            merged_df = pd.merge(merged_df, lime_gen_times_df, on=['dataset', 'algorithm', 'seed'], how='left')
        elif 'LIME Values Time (s)' not in merged_df.columns:
            merged_df['LIME Values Time (s)'] = np.nan

        if not shap_gen_times_df.empty and all(col in shap_gen_times_df.columns for col in ['dataset', 'algorithm', 'seed']):
            merged_df = pd.merge(merged_df, shap_gen_times_df, on=['dataset', 'algorithm', 'seed'], how='left')
        elif 'SHAP Values Time (s)' not in merged_df.columns:
            merged_df['SHAP Values Time (s)'] = np.nan

        timing_cols_to_average = [
            'LIME Explainer Time (s)', 'LIME Values Time (s)',
            'SHAP Explainer Time (s)', 'SHAP Values Time (s)'
        ]

        for col in timing_cols_to_average:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            else:
                merged_df[col] = np.nan

        mean_timings = merged_df.groupby(['dataset', 'algorithm'])[timing_cols_to_average].mean().reset_index()

        self.timing_summaries_by_dataset = {}
        datasets = mean_timings['dataset'].unique()

        for ds_name in datasets:
            dataset_df = mean_timings[mean_timings['dataset'] == ds_name].copy()
            dataset_df = dataset_df.drop(columns=['dataset'])
            dataset_df = dataset_df.set_index('algorithm')

            final_cols_order = [
                'LIME Explainer Time (s)', 'LIME Values Time (s)',
                'SHAP Explainer Time (s)', 'SHAP Values Time (s)'
            ]

            dataset_df = dataset_df[[col for col in final_cols_order if col in dataset_df.columns]]
            self.timing_summaries_by_dataset[ds_name] = dataset_df

            summary_file_path = self.results_base_dir / f"{ds_name}_timing_summary.csv"
            try:
                title_header = f"Dataset: {ds_name}\n"
                with open(summary_file_path, 'w') as f:
                     f.write(title_header)
                     dataset_df.to_csv(f, float_format='%.4f')
                logging.info(f"Saved timing summary for dataset '{ds_name}' to: {summary_file_path}")
            except Exception as e:
                logging.error(f"Error saving timing summary for dataset '{ds_name}': {e}", exc_info=True)

        logging.info("Finished timing summary table calculations.")


if __name__ == "__main__":
    analyzer = MetricsAnalyzer(results_base_dir="results", top_n_features=10)

    logging.info("Running LIME feature name fixer...")
    analyzer.fix_lime_feature_names()
    logging.info("LIME feature name fixer finished.")

    analyzer.calculate_mean_metrics_per_run()
    analyzer.calculate_overall_mean_metrics()
    analyzer.calculate_and_save_per_experiment_agreement()
    analyzer.calculate_and_save_overall_agreement_tables()
    analyzer.calculate_and_save_timing_summary_tables()

    logging.info("Analysis complete.")

