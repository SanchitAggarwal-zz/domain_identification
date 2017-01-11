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

For simultaneously training and testing and saving results, run:
``` sh
python di_main.py -t data/train.tsv -m model.pkl -i data/test.txt -r resutls.tsv
```


### Pre-processing ###
Performed pre-processing of messages using NLTK. Function `messageToWords` takes a message as input, it then  removes html markup, stop words, any punctuation and lemmatize the tokens in the message.

``` python
# Function to convert a raw message to a string of words
# The input is a single string (a raw message), and
# the output is a single string (a preprocessed message)
def messageToWords( message ):
    message_text = BeautifulSoup(message,"html.parser").get_text() # remove html
    clean_message = re.sub("[^a-zA-Z]", " ", message_text) # remove non-letters
    words = clean_message.lower().split() # convert to words
    words = [w for w in words if not w in stopwords]
    words = [w for w in words if not w in punctuation]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if minlength < len(w) < maxlength]
    return( " ".join( words ))
```


### Experiments ###
Performed different experiments for feature selection and classifier selection. For all the experiments we divided the training data into training set and validation set with a validation set size of 0.3

``` python
    # split into training and validation set
    print "splitting data into training and validation set"
    training_set, validation_set = train_test_split(clean_training_df, test_size = 0.3)
    print training_set.shape
    print validation_set.shape
```

#### **Experiment 1** *Bag of Words - Naive Bayes - 30% Validation* ####
For this, we use Bag of Words representation for each message and use a Naive Bayes Classifier.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**shopping**   |0.94      |0.88      |0.91       |455|
|**food**       |0.95      |0.72      |0.82       |327|
|**travel**     |0.87      |0.81      |0.84       |349|
|**reminders**  |0.97      |0.85      |0.91       |153|
|**movies**     |0.98      |0.94      |0.96       |568|
|**nearby**     |0.85      |0.98      |0.91      |1295|
|**support**    |0.84      |0.61      |0.71       |132|
|**recharge**   |0.91      |0.93      |0.92       |902|
|**avg / total**|**0.91**      |**0.90**      |**0.90**      |**4181**|

#### **Experiment 2** *TFIDF - Naive Bayes - 30% Validation* ####
For this, we use TFIDF representation for each message and use a Naive Bayes Classifier.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**shopping**   |0.97     | 0.80     | 0.87   |    422|
|**food**       |0.97     | 0.61     | 0.75   |    340|
|**travel**     |0.95     | 0.27     | 0.43   |    142|
|**reminders**  |0.99     | 0.71     | 0.83   |    170|
|**movies**     |0.97     | 0.93     | 0.95   |    552|
|**nearby**     |0.79     | 0.99     | 0.88   |   1335|
|**support**    |0.87     | 0.84     | 0.86   |    330|
|**recharge**   |0.88     | 0.92     | 0.90   |    890|
|**avg / total**|**0.88**     | **0.87**     | **0.86**   |   **4181**|

#### **Experiment 3** *TFIDF - SVM - 30% Validation* ####
For this, we use TFIDF representation for each message and use a SVM Classifier.
*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**shopping**   |0.93     |0.83    | 0.88      | 433|
|**food**       |0.93     |0.78    | 0.85      | 341|
|**travel**     |0.89     |0.85    | 0.87      | 328|
|**reminders**  |0.94     |0.84    | 0.89      | 135|
|**movies**     |0.96     |0.94    | 0.95      | 543|
|**nearby**     |0.87     |0.96    | 0.92      |1342|
|**support**    |0.81     |0.66    | 0.73      | 139|
|**recharge**   |0.88     |0.91    | 0.89      | 920|
|**avg / total**|**0.90** |**0.90**| **0.90**  |**4181**|


#### **Experiment 3** *TFIDF - RF- 30% Validation* ####
For this, we use TFIDF representation for each message and use a Random Forest Classifier.

*Results*:

|               |precision|  recall|  f1-score|  support|
| ------------- |:-------:| ------:| --------:|--------:|
|**shopping**   |   0.97  |    0.85|      0.90|       400|
|**food**       |   0.93  |    0.75|      0.83|       336|
|**travel**     |   0.90  |    0.84|      0.87|       340|
|**reminders**  |   0.95  |    0.87|      0.91|       167|
|**movies**     |   0.97  |    0.95|      0.96|       523|
|**nearby**     |   0.87  |    0.96|      0.91|      1379|
|**support**    |   0.78  |    0.70|      0.74|       123|
|**recharge**   |   0.89  |    0.91|      0.90|       913|
|**avg / total**| **0.90**|**0.90**|  **0.90**|  **4181**|

#### **Conclusion** ###
