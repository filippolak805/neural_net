# neural_net

Fashion-MNIST classifier in C++20 (from scratch, no ML libraries).

## Requirements

- g++ with C++20 support
- gzip

## Run

```bash
./run.sh
```

Extracts the dataset (`prepare_data.sh`), compiles to `./network`, and runs training.
Outputs: `test_predictions.csv`, `train_predictions.csv`.

## Debug

Open in VS Code with the C/C++ extension installed and press F5.

## Dataset

Plain CSVs in `data/` are git-ignored. After changing them, re-run `./compress_data.sh` and commit the `.csv.gz` files.
