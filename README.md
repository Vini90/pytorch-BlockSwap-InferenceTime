## code layout
- `checkpoints/` is used to save trained models
- `genotypes/` is used to store `.csv` files that contain network configurations chosen by Inference time based BlockSwap
- `archs/` can be used to save large dataframes containing pre-sorted random configurations. For example, we could randomly generate architectures, get their inference time, and save them. Then when we go to do Fisher ranking we can quickly get random samples by indexing the dataframe to return rows that satisfy our time budget
- `models/` contains PyTorch definitions for all of the models and blocktypes that the Block-swap code used
    - `models/blocks.py` is where all of the block substitutions live
- `count_ops.py` contains basic model measurement functions
- `funcs.py` contains useful operations that are used throughout the code execution. It also includes random configuration sampling code.
    - `one_shot_fisher` is the function used to get the Fisher potential of a given network
    - `cifar_random_search` writes a dataframe of random configs to `archs/` to later be Fisher-ranked
- `fisher_rank.py` ranks random configurations at a given time goal
- `main.py` can train your selected network

## Running the experiments
First, train a teacher network on a dataset of your choice. For example, to use CIFAR-10:
```bash
python main.py cifar10 teacher --conv Conv -t wrn_40_2_1 --wrn_depth 40 --wrn_width 2 --cifar_loc='<path-to-data>' --GPU '0,1'
```

The next step is to generate a dataframe of random network configurations, then set some parameter goal and sample:
```
python fisher_rank.py cifar10 --generate_random
python fisher_rank.py cifar10 --data_loc='<path-to-data>' --inference_time $t
```
This will save `.csv` files each time a new "best" model is found. Train the highest numbered genotype using:
```
python main.py cifar10 student --conv Conv -t wrn_40_2 -s wrn_40_2_<genotype-num> --wrn_depth 40 --wrn_width 2 --cifar_loc='<path-to-data>'  --GPU 0 --from_genotype './genotypes/<genotype-num>.csv'
```

## Acknowledgements
https://arxiv.org/abs/1906.04113
https://github.com/BayesWatch/pytorch-blockswap


