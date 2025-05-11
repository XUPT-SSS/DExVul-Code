# DExVul

## Environment Setup
- Setup Python environment:
```
python 3.10.15
```

- Setup Python Package:
```
pip install -r requirements.txt
```

## Command Usages
#### Args usages
- `--from_pretrained`: `Salesforce/codet5-small`, `Salesforce/codet5-base`
- `--dataset`: `ctf`, `magma`, `MegaVul`, `Big-Vul`, `Devign`, `Reveal`
- `--small_sample`: `_25`, `_50`, `_75`
- `--alpha`: Task weight for multi-task training. Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss
  - `--alpha 0.5`: recommended
- `--batch_size`: Batch size
- `--max_input_length`: Maximum input length
- `--lr`: Learning rate. Default is 5e-6.
- `--epochs`: Maximum epoch for training. Default is 20.
- `--model_type`:
  - `standard`: Standard finetuning
  - `task_prefix`: DExVul


#### Example usages
- Standard finetuning with `Full datasets`:
```python
python run.py --from_pretrained Salesforce/codet5-base --dataset MegaVul --model_type standard --batch_size 16
```


- DExVul with `Full datasets` and `ChatGPT rationale`:
```python
python run.py --from_pretrained Salesforce/codet5-base --dataset MegaVul --model_type task_prefix --alpha 0.5 --batch_size 16
```
If you want to run small sample training, please add `--small_sample _x` to the command. (`x` represents the number of samples, selecting from 25, 50, 75)