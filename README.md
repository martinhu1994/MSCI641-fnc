# MSCI641-fnc
This is the repository for MSCI641 final project - Fake News Challenge  
Repo Link: https://github.com/martinhu1994/MSCI641-fnc

## Team Member
Name: Hanxiao Hu  
Student ID: 20767041  
E-mail: h73hu@uwaterloo.ca

## Project Structure
There are two main directories: **src** and **data**  
```
src:
    - model.py
    - tokenizer.py
    - score.py
data:
    - train_stances.csv
    - train_bodies.csv
    - competition_test_stances.csv
    - competition_test_bodies.csv
```
## Prerequisite (\*IMPORTANT\*)
Before running the project, two steps need to be done.
1. Please follow the link [GloVe](https://nlp.stanford.edu/projects/glove/) to download the pre-trained embedding. Under the *Download pre-trained word vectors* section, the **glove.6B.zip** file is desired. Unzip the file after the completion of download and place the file **glove.6B.300d.txt** into ```data``` directory. 
2. Run the **tokenizer.py** script by issuing ```$ python toknizer.py``` command. It will generate two data files: **processed_data.csv** and **processed_test.csv**. These two files are placed in ```data``` directory automatically. 

## Execution
### Execution Command
Please choose one of the commands to run the program. The default model will give the best result.
1. ```$ python model.py```  
This command will execute the default model *Bidirectionl Conditional Encoding Model*.
2. ```$ python model.py 1```  
This command will execute the first model *Parallel Encoding Model*.  
3. ```$ python model.py 2```  
This command will execute the second model *Conditional Encoding Model*.  
### Leaderboard Score Reproduction Issue
My reported best score on Codalab leaderborad is 7488.25. This score comes from default model. However, it is hard to reproduce this score excatly. The problem comes from the random split of training data and validation data. Everytime the split may be different and this difference may cause the performance fluctuation. Another factor affecting the score is the number of training epoches. The given epoch numbers in the code is just for reference, which gives the best score during my experiment. In order to achieve the best peformance, please adjust the number of epoches to control the training loss around 0.025 for default *Bidirectional Conditional Encoding Model*. For the other two models, the first model will be fine with the given epoches. The second model performs the best when the trainning loss is controlled around 0.050.

## Output
The program will print the score on screen when it finishes. A **submission.csv** file will be placed in ```data``` directory. This file is the final prediction.
