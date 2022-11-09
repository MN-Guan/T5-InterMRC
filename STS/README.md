# Semantic Textual Similarity (STS) Module 
## Directory Guide
```
| -- **STS**  
    | -- Data                  # Save training datasets and model parameters  
    | -- __init__.py           # Identifier of package  
    | -- STS_preprocess.py     # Semantic Text Similarity Preprocessing Function  
    | -- T5.py                 # Training Script   
    | -- Test.py               # Sample test script for later annotation evidence  
```

## Training
Switch to the path where the `T5.py` script is located, then
```bash
python T5.py
```
See [T5.py](https://github.com/MN-Guan/T5-InterMRC/blob/master/STS/T5.py) script for more hyper-parameter modification.
## Results
During training, the cached file named `STSb_no_padding_cached_0.1_128_aug_div.json` will be generated. The cache file retains the preprocessed training data, so that the model can be directly loaded for use in the future without repeated preprocessing every time, unless the preprocessing operation changes.

After the training, `loss.txt` file will be generated in the path where the `T5.py` script is located, which shows the performance of the model in the dev set at all checkpoints. Finally, we select the model that performed best on the development set to obtain the performance on the test set. The last line of `loss.txt` file shows the performance of the model in the test set.

The parameter file of the best model is displayed in [Data/Model_State](https://github.com/MN-Guan/T5-InterMRC/tree/master/STS/Data/Model_States) directory.

Subsequently, we will use the STS model to label the evidence. See [Reconstruct_Data.py](https://github.com/MN-Guan/T5-InterMRC/tree/master/MRC/Reconstruct_Data.py)
