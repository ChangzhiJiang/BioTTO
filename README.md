# BioTTO

BioTTO is designed for test-time optimization and generation of peptide (AMP) and DNA sequences.

## Environment Setup

The AMP environment can be created using `amp/environment.yml`:

```bash
conda env create -f amp/environment.yml
conda activate peptidedto
```

Before running the code, prepare the base model, training data, and the predictor, prior, tokenizer, and other files referenced in the configuration files. Replace `/path/to/...` in the commands below with the actual paths.

## AMP Usage

Enter the AMP directory:

```bash
cd amp
```

Train the LoRA adapter:

```bash
python scripts/train_sft_from_config.py \
  --config configs/amp.yaml \
  --base-model /path/to/base_model \
  --data-csv /path/to/data.csv \
  --output-dir checkpoints/lora/amp \
  --device cuda:0
```

Generate peptide sequences using the trained LoRA checkpoint:

```bash
python scripts/generate.py \
  --config configs/amp.yaml \
  --base-model /path/to/base_model \
  --lora-checkpoint /path/to/lora_checkpoint \
  --data-csv /path/to/data.csv \
  --output-dir outputs/amp \
  --device cuda:0
```

To use only the base model, add `--without-sft` to the generation command and omit `--lora-checkpoint`.

## DNA Usage

The DNA module requires PyTorch, Evo2 model weights, and an activity predictor checkpoint. Enter the DNA directory:

```bash
cd dna
```

First, generate a fixed pool of initial sequences:

```bash
python scripts/sample_initial_sequences.py \
  --output outputs/initial_sequences.json \
  --length 200 \
  --num-sequences 32
```

Then run the search:

```bash
python scripts/run_search_sweep.py \
  --design-checkpoint /path/to/design_checkpoint \
  --initial-sequences outputs/initial_sequences.json \
  --configs configs/dna_tto_uncapped_hepg2_v1.json \
  --output-dir outputs/search
```

By default, the search runs for `K562`, `HepG2`, and `SK-N-SH`. To run the search for only one target, add an option such as `--targets HepG2`.

## Command Help

To view the AMP script help messages:

```bash
cd amp
python scripts/train_sft_from_config.py --help
python scripts/generate.py --help
```

To view the DNA script help messages:

```bash
cd ../dna
python scripts/sample_initial_sequences.py --help
python scripts/run_search_sweep.py --help
```

For more complete experimental settings, refer to the configuration files in `amp/configs/` and `dna/configs/`.
