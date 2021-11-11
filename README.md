# Deep-learning-project2
This is for the deep learning course project-2

## Descriptions
1. As instructed, there is a directory hw2/hw2_1 which contains 1). report.pdf, 2). seq2seq_model (./saved_model) 3. hw2_seq2seq.sh, and 4).model_seq2seq.py

2. The model can be directly downloaded from google drive via the shell script

3. The model in hw2/hw2_1 is the seq2seq+attention+schedule_sampling

4. Run the sh file (example): ./hw2_seq2seq.sh ./feature_dirs_testing ./generated_video_caption.txt

4. The Bleu score is 0.6764 

5. The test and training data can be download from https://drive.google.com/open?id=1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ, training feature files can be put into ./feature_dirs_training. The caption files can be put into ./captions

6. In the hw2/others, there are seq2seq, seq2seq+attention, seq2seq+schedule_sampling models


## Required packages
1. python 3.6
2. tensorflow 1.14
3. numpy 1.19
4. pandas 1.1.5
