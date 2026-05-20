# K1-POI

The official implementation of K1-POI for the KDD 2026 paper **Is the Last Check-In All You Need? Next POI Recommendation: Recall and Rerank**.

## Quick Start

To set up the environment, run the following commands:

```bash
conda create -n next-poi python=3.12 -y
conda activate next-poi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

This step is flexible. You can install different versions of Python and PyTorch using other environment management tools.

Clone the repository and use `k1-poi` as the working directory:

```bash
git clone https://github.com/XZJIsme/k1-poi.git
cd k1-poi
tar -xzf data/raw/raw.tar.gz -C data/raw
python data/data_process.py
```

These commands unpack `data/raw/raw.tar.gz` into:

- `data/raw/Gowalla-CA`
- `data/raw/NYC`
- `data/raw/TKY`

Then `python data/data_process.py` processes these raw datasets and writes the processed pickle files to `data/processed`.

To run the training and evaluation for K1-POI, run the following commands. The command uses `nohup` to allow it to run in the background, and redirects the output to a log file in `exp_scripts/nohup_logs/`. You can adjust the command-line arguments as needed. The command will run the three datasets all together with `--just_run_all_runs_together` on multiple devices which are specified by `--devices` in parallel, but if you want to run them sequentially, you can remove that flag. You may modify `--devices` to specify the GPU devices you want to use (e.g., `--devices 0,1` to use GPU 0 and 1, or `--devices 0` to use only GPU 0). If you want to run only one dataset, you can specify only one dataset in `--data_path_list`. If you want to run on the REPLAY datasets (see below), you should change the dataset paths in `--data_path_list` accordingly. Since the REPLAY datasets are larger, you may also want to adjust `--search_trials` to smaller values for accelerating the hyperparameter search.

The script will produce a markdown file in `exp_scripts/results/`. The markdown report starts with summary tables for each dataset and window size, comparing the test metrics without the reranker (`w/o reranker`) and with the reranker (`w/ reranker`). After the summary tables, the report includes detailed results for each dataset.

Run on the three datasets together across multiple devices in parallel:

```bash
mkdir -p exp_scripts/nohup_logs
nohup python -u exp_scripts/run_st_prior_reranker_hparam_search_parallel_trials.py \
  --data_path_list data/processed/TKY_excluding_cold.pkl \
  data/processed/NYC_excluding_cold.pkl \
  data/processed/CA_excluding_cold.pkl \
  --epochs 100 \
  --window_size_list 1 \
  --devices 0,1 \
  --self_attn_d_model 128 \
  --self_attn_num_layers 2 \
  --self_attn_num_heads 4 \
  --poi_ecl_negative_source topk \
  --st_prior_candidate_k 200 \
  --search_trials 20000 \
  --search_eval_samples 20000 \
  --search_top_m 400 \
  --st_prior_num_parallel_trials 20 \
  --just_run_all_runs_together \
  > exp_scripts/nohup_logs/st_prior_reranker_hparam_search_parallel_trials_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Run on NYC only using GPU 0:

```bash
mkdir -p exp_scripts/nohup_logs
nohup python -u exp_scripts/run_st_prior_reranker_hparam_search_parallel_trials.py \
  --data_path_list data/processed/NYC_excluding_cold.pkl \
  --epochs 100 \
  --window_size_list 1 \
  --devices 0 \
  --self_attn_d_model 128 \
  --self_attn_num_layers 2 \
  --self_attn_num_heads 4 \
  --poi_ecl_negative_source topk \
  --st_prior_candidate_k 200 \
  --search_trials 20000 \
  --search_eval_samples 20000 \
  --search_top_m 400 \
  --st_prior_num_parallel_trials 20 \
  --just_run_all_runs_together \
  > exp_scripts/nohup_logs/st_prior_reranker_hparam_search_parallel_trials_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

An example command to run on the three datasets together across multiple devices in parallel with all CLI parameters explicitly specified:

```bash
mkdir -p exp_scripts/nohup_logs
nohup python -u exp_scripts/run_st_prior_reranker_hparam_search_parallel_trials.py \
  --data_path_list data/processed/TKY_excluding_cold.pkl \
  data/processed/NYC_excluding_cold.pkl \
  data/processed/CA_excluding_cold.pkl \
  --epochs 100 \
  --window_size_list 1 \
  --devices 0,1,2,3 \
  --self_attn_d_model 128 \
  --self_attn_num_layers 2 \
  --self_attn_num_heads 4 \
  --use_cat_emb false \
  --use_user_emb true \
  --user_emb_dim 256 \
  --poi_emb_dim 128 \
  --use_positional_encoding true \
  --use_tod_slot_embedding true \
  --tod_slot_scales 6 12 24 \
  --tod_slot_emb_dim 64 \
  --use_geo_cell_embedding true \
  --geo_cell_sizes_m 500 1000 2000 \
  --geo_cell_emb_dim 64 \
  --use_poi_embedding_contrastive_learning true \
  --top_k_candidates_for_poi_embedding_contrastive_learning 200 \
  --use_mlp_for_cl_instead_of_simple_proj false \
  --poi_embedding_contrastive_learning_force_label_into_candidates_strategy replace_lowest \
  --poi_embedding_contrastive_learning_proj_dim 128 \
  --poi_embedding_contrastive_learning_temperature 0.07 \
  --poi_embedding_contrastive_learning_normalize_embeddings true \
  --poi_embedding_contrastive_learning_loss_weight 1.0 \
  --poi_ecl_negative_source topk \
  --st_prior_candidate_k 200 \
  --st_prior_time_bins 24 \
  --st_prior_alpha 1.0 \
  --st_prior_user_bin_min_count 5 \
  --search_trials 20000 \
  --search_eval_samples 20000 \
  --search_top_m 400 \
  --st_prior_num_parallel_trials 20 \
  --st_prior_score_terms time user dist cl_sim \
  --just_run_all_runs_together \
  > exp_scripts/nohup_logs/st_prior_reranker_hparam_search_parallel_trials_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Data Citation

If you use `k1-poi/data/raw/raw.tar.gz`, please cite the [GETNext paper](https://dl.acm.org/doi/10.1145/3477495.3531983).

The Gowalla and Foursquare datasets from REPLAY are not the main datasets used in our paper. Note that the REPLAY Gowalla dataset is different from the `Gowalla-CA` dataset included in `data/raw/raw.tar.gz`.

To download the REPLAY data, run the following commands from the `k1-poi` directory:

```bash
wget -O data/raw/data.zip "https://www.dropbox.com/s/6qyrvp1epyo72xd/data.zip?dl=1"
mkdir -p data/raw/replay-4sq-gwl
unzip data/raw/data.zip -d data/raw/replay-4sq-gwl
python data/data_process_4_replay_4sq_gwl.py
```

The REPLAY processing command reads `data/raw/replay-4sq-gwl/data` and writes the processed pickle files to `data/processed`.

If you use the REPLAY Gowalla or Foursquare datasets, please cite their [paper](https://ieeexplore.ieee.org/document/10971252/).
