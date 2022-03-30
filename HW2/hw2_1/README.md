# HW2
This is the second homework for the CPSC8200 Deep Learning class. Here we train the Seq2Seq model. 
To run the python script following packages needs to be installed :

-PyTorch
-Numpy
-TQDM

In the repo, you will find index2word.pickle, word2index.pickle and s2vt, please keep these files in the same directory as the 
script for loading the vocab and model state. The model is provided as s2vt.
To Run the script.
```
python ./modelseq2seq.py ./data_directory/ output_file_name.txt
```