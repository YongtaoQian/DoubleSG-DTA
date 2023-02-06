“DoubleSG-DTA: Deep Learning for Drug Discovery: Case Study on The Non-Small Cell Lung Cancer with EGFR T790M Mutation”

step 1:Environmental settings
pytorch=1.11.0
python=3.8.0
cuda=11.3


step 2:
create_data.py: The input data (SMILES \t FASTA squence \t label), the purpose is to convert SMILES sequences and FASTA sequences into integer vectors and then into word embedding, in addition to converting SMILES sequences into a response drug graph.
utils.py:Loading data sets to provide evaluation metrics calculations
training.py:Training the model and recording the results data
ginconv.py:pytorch code implementation of the DoubleSG-DTA model


step3:run code
1.python xxxxxx/xxxxx/load_data.py
2.python xxxxx/xxxxxx/DoubleSG-DTA_Train_main.py num1,num2   (num1:num1 means select the data set, either 1,2,3,  num2:num2 indicates the number of cuda selected)

We currently upload only the training and test sets to simplify the model run time, so that you can train the DoubleSG-DTA model directly and obtain the prediction performance from the test set.

If you want to use our model to process your dataset, you will need to format the dataset as follows:

compound_iso_smiles     target_sequence     affinity
CCCCC(c1ccc)            NWCVQIA               12.4



We have provided the code for the case study 1， 2， 3 section of the article.

case study :the effect of the number of GIN layers
case study :the effect of SEnet
case study :the effect of Coss Multi-head attention 

Due to the rush, the full version of the code was stored on the server and we put together a model with a training set, test set, validation set, and well-trained weights as soon as possible.
