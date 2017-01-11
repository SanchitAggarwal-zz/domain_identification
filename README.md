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
python  python di_main.py -t data/train.tsv -m model.pkl -i data/test.txt
```


### Experiments ###
