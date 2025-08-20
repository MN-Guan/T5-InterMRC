# Machine Reading Comprehension (MRC) Module 
## Directory Guide
```
| -- **MRC**  
    | -- Data                  # Save training data and model parameters  
    | -- MQA_preprocess.py     # Machine Reading Comprehension Preprocessing script  
    | -- Prediction.py         # Use saved MRC model to predict answers and evidences and generate prediction file  
    | -- Reconstruct_Data.py   # Script file for labeling evidence in SQuAD dataset using STS model    
    | -- T5InterMRC.py         # Training Script for T5-InterMRC     
    | -- BartInterMRC.py       # Training Script for Bart-InterMRC
    | -- eval_expmrc.py        # ExpMRC official evaluation script
    | -- loss_bart-base2.txt   # The file to record the training log of Bart-InterMRC model 
    | -- loss_t5-large.txt     # The file to record the training log of Bart-InterMRC model 
    | -- run.sh                # The script for train our MRC models
```
## Evidence annotation
Switch to the path where the `Reconstruct_Data.py` script is located.

Then, simply run the following command:
```
python Reconstruct_Data.py
```
or
```
python Reconstruct_Data.py \
--random_seed=52 \
--model_name=t5-base \
--batch_size=50 \
--state_file=${sts_state_file} \
--device=cuda:0 
```
- `model_name`: Name of basic STS model used to label evidence (Consistent with the model in the `state_file`)
- `state_file`: The parameter file of trained STS model

The dataset with annotated evidence will be shown in `MRC/Data/Datasets/ExpMRC/`

## Training
Switch to the path where the `T5InterMRC.py` script is located. 

Then, simply run the following command:
```
python T5InterMRC.py --is_new --is_training
```
or
```bash
python T5InterMRC.py \
--random_seed=52 \
--model_name=T5-InterMRC \
--model_level=t5-large \
--left_threshold=0.01 \
--right_threshold=0.3 \
--N_EPOCHS=10 \
--max_len=500 \
--batch_size=200 \
--mini_batch_size=2 \
--warm_up_rate=0.1 \
--learning_rate=1e-3 \
--weight_decay=0.0 \
--clip_threshold=1.0 \
--save_steps=100 \
--data_name=squad \
--data_path=./Data/Datasets/ExpMRC/ \
--device=cuda:0 \
--log_file_name=./loss.txt \
--state_file=./Data/T5${time}.pt \
--prediction_file=./pred.json \
--mode=t5-base \
--is_new \
--is_training
```
- `model_name`: Name of the model you are using (must be one of the 'T5-Shared', 'T5-InterMRC'(default) and 'T5-Independent')
- `model_level`: The name of basic T5 model ('t5-base', 't5-large', ...), defaulting to 't5-large'
- `log_file_name`: Record the dev set results of all checkpoints
- `state_file`: We use time to recognize the model. Such as `T5_11-3-17.pt`
- `mode`: Mode of annotating evidence. `t5-base` denotes that `t5-base` model is used to annotate evidence
- `is_new` indicates whether to use the newly labeled data to train the model. If it is set to false, it indicates to use pseudo data to train the model

## Prediction and Evaluation
After training, you can use `Prediction.py` to get the prediction file by running:
```
python Prediction.py
```
You can customize the following arguments:
- `model_name`: Name of the model you are using (must be one of the 'T5-Shared', 'T5-InterMRC'(default) and 'T5-Independent')
- `model_level`: The name of basic T5 model ('t5-base', 't5-large', ...), defaulting to 't5-large'
- `state_file`: Full directory of model parameter file
- `batch_size`: The batch size of dev process, defaulting to 50
- `dev_file_name`: Full directory of dev dataset file, defaulting to './Data/Datasets/ExpMRC/expmrc-squad-dev.json'
- `prediction_file`: Full directory of model prediction file, defaulting to './pred.json'

After obtaining the prediction file (named 'pred.json' by default), you can use official evaluation script to get the results by running:
```
python eval_expmrc.py ${dev_file_name} ${prediction_file}
```
`dev_file_name` and `prediction_file` are described as above. Simply, you can run the following command:
```
python eval_expmrc.py ./Data/Datasets/ExpMRC/expmrc-squad-dev.json ./pred.json
```

## Results
During training, the cached file named `padding_t5-base_squad_500.json` will be generated. The cache file retains the preprocessed training data, so that the model can be directly loaded for use in the future without repeated preprocessing every time, unless the preprocessing operation changes.

After training, `loss.txt` file will be generated in the path where the `T5InterMRC.py` script is located, which shows the performance of the model in the dev set at all checkpoints.

The parameter file of the best model will be displayed in `Data/` directory.
