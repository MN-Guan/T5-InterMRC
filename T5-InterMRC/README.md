# T5-InterMRC
Code for the paper "A T5-based Interpretable Reading Comprehension Model with more accurate evidence training"

# Directory Guide
```
| -- **T5-InterMRC Root**  
    | -- MRC                     # Machine Reading Comprehension Module  
    | -- STS                     # Semantic Textual Similarity Module  
    | -- __init__.py             # Identifier of package  
    | -- Auxiliary_Functions.py  # Describe auxiliary functions such as display of output and initialization of random variables   
    | -- Metrics.py              # Metrics script for various tasks  
    | -- Preprocess.py           # Common preprocessing script for each task  
    | -- T5Model.py              # Script of T5-InterMRC model    
    | -- BartModel.py            # Script of Bart-InterMRC model  
```
# Requirements
```
python 3.8.12
```

# Install dependent packages
We use `Pytorch` open source framework to build our model. You can use the following two methods to build the development environment:

- Method 1: Using pip instruction
```bash
pip install -r requirements.txt 
```
- Method 2: Using docker environment
```bash
boxuguan/deeplearning 
```
See [Docker](https://hub.docker.com/repository/docker/boxuguan/deeplearning) for more details
# Contact us
Please submit an issue or sent an email to boxv985701@163.com.
