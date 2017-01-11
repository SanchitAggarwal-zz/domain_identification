## **Domain Identification** ##

To identify the domain of the message for automatic conversational agents.

### Datasets ###
Find the dataset [here](https://github.com/SanchitAggarwal/domain_identification/tree/master/data). There are two files:

 1. [train.tsv](https://github.com/SanchitAggarwal/domain_identification/blob/master/data/train.tsv) : Each line contains a message and corresponding domain.
 2. [test.txt](https://github.com/SanchitAggarwal/domain_identification/blob/master/data/test.txt) : Each line is a separate message.

### Requirements ###

 - python 2.7
 - python modules:
	 - nltk
	 - scikit-learn
	 - beautifulsoup4
	 - pandas

### The Code ###
Clone the repository.

``` sh
git clone https://github.com/SanchitAggarwal/domain_identification.git
cd domian_identification
```

For training the model, run:
``` sh
python  di_main.py -t data/train.tsv
```

This will save a `model.pkl` file at the root folder.

For testing the model, run:
``` sh
python  di_main.py -m model.pkl -i data/test.txt
```

For simultaneously training and testing, run:
``` sh
python di_main.py -t data/train.tsv -m model.pkl -i data/test.txt
```


### Experiments ###
Performed different experiments for feature selection and classifier selection. I divided the training data into training set and validation set with a validation set size of 0.3

``` python
    # split into training and validation set
    print "splitting data into training and validation set"
    training_set, validation_set = train_test_split(clean_training_df, test_size = 0.3)
    print training_set.shape
    print validation_set.shape
```

 1. **Experiment 1** *Bag of Words - Naive Bayes - 30% Validation*
For this, we use Bag of Words representation for each message and use a Naive Bayes Classifier. We split the data, so that 30% of messages are used for validation.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**shopping**   |0.96     |0.81    |0.88      |431      |
|**food**       |0.96     |0.63    |0.76      |352      |
|**travel**     |0.90     |0.84    |0.87      |341      |
|**reminders**  |1.00     |0.67    |0.80      |156      |
|**movies**     |0.96     |0.93    |0.95      |525      |
|**nearby**     |0.88     |0.91    |0.90      |893      |
|**support**    |0.90     |0.27    |0.42      |136      |
|**recharge**   |0.79     |0.99    |0.88      |1347     |
|**avg / total**|**0.88**     |**0.87**    |**0.86**     |**4181**     |
