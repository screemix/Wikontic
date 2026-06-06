## How to run distillation
First change directory to distillation:
```
cd distillation
```
1. Prepare data:<br>
You can change HOTPOT_PATH and DUMP_PATH in split_data.py
<br>
then run:
```
python split_data.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
or
python build_combined_dataset.py  --source hotpot200:datasets/hotpotqa200.json:datasets/kg_dump_hotpot_gpt4_1_onto_triplets.json --source musique_gpt4:datasets/musique_200_test.json:datasets/kg_dump_musique_gpt4_1_onto_triplets.json   --triplet-field triplets --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
2. To train

```
python train.py --config configs/train.yaml
```

3. To infer (evaluate)<br>
You can choose either _val_ or test _split_
```
python infer.py --config configs/infer.yaml --split test
```
