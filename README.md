# Repository for Take home exam: Challenging SRL
## Meru Nurberdikhanova, 2779728
## Structure
This repository contains the following files:
- ```main.py```: that contains all the code for creation, testing, and evaluation of challenge sets
- ```requirements.txt```: that contains the dependencies needed to be installed to run ```main.py```
- ```data```: that contains 17 files, 8 of which are .jsonl challenge sets with gold labels, 8 are .jsonl files with model predictions, and 1 is a .tsv file with failure rates of the models on challenge sets

## How to run ```main.py```
- ```pip install -r /path/to/requirements.txt``` to install the dependencies needed
- scroll down the ```main.py``` and specify your arguments: boolean_1 is whether to create the challenge sets, boolean_2 is whether to make model predictions over the sets, and boolean_3 is whether to calculate the failure rates on the sets
- you are done! great work, now you can inspect the hell of the code and datasets I have made~