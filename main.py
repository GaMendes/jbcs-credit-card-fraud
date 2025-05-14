import pandas as pd
import time
import logging

from data_loader import load_data
from model_wrapper import ModelWrapper
from config import get_model_params, get_models_to_run, get_datasets_to_run
from utils import (ResultsSaver, create_results_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_all_experiments():

    """Runs the full experimental pipeline for configured datasets, models, and seeds."""

    datasets = get_datasets_to_run()
    models_config = get_models_to_run()
    seeds = range(1, 6)

    overall_start_time = time.time()

    for seed in seeds:

      for dataset_name in datasets:
        logging.info(f"--- Processing Dataset: {dataset_name} ---")
  
        try:
            X, y = load_data(dataset_name)
            if not isinstance(X, pd.DataFrame) or X.columns.empty:
                raise ValueError("Loaded data X is not a DataFrame or has no columns.")
    
        except FileNotFoundError as e:

            logging.error(f"Dataset file not found for '{dataset_name}': {e}. Skipping dataset.")
            continue

        except Exception as e:
            logging.error(f"Error loading/validating dataset '{dataset_name}': {e}. Skipping dataset.", exc_info=True)
            continue

        for model_name, model_instance in models_config.items():
          logging.info(f"--- Running Model: {model_name} ---")
          params = get_model_params(model_name, seed)

          logging.info(f"--- Running Experiment with Seed: {seed} ---")
          experiment_name = f"experiment{seed}"
          results_dir = create_results_dir(experiment_name, dataset_name, model_name)

          run_start_time = time.time()

          try:
            model_wrapper = ModelWrapper(model_instance, params, random_seed=seed)
            results = model_wrapper.train(X, y, dataset_name)

            if not results or not results.get('models'):
                logging.warning(f"Model training produced no results for model '{model_name}', seed {seed}, dataset '{dataset_name}'. Skipping saving.")
                continue

            saver = ResultsSaver(results, seed, results_dir, model_name)

            if (model_name == "LogisticRegression"):
              saver.save_ground_truth()

            saver.save_metrics()
            saver.save_explanations()
            saver.save_timings()

          except Exception as e:
            logging.error(f"Critical error during run for model '{model_name}', seed {seed}, dataset '{dataset_name}': {e}", exc_info=True)
            continue

          run_end_time = time.time()
          logging.info(f"--- Finished Experiment processing for Seed: {seed} in {run_end_time - run_start_time:.2f} seconds ---")

          # End of model loop
        # End of dataset loop for this dataset
      # End of seeds loop

    overall_end_time = time.time()
    logging.info(f"--- Finished all experiments in {overall_end_time - overall_start_time:.2f} seconds ---")


if __name__ == "__main__":
    run_all_experiments()
