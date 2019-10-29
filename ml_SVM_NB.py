import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def classify(training_file, testing_file):

    # import training file and testing file
    training_file = pd.read_csv(training_file)
    testing_file = pd.read_csv(testing_file)

    # build a pipeline simulating the steps (1) generate tfidf (2)train SVM
    text_clf = Pipeline(
        [("tfidf", TfidfVectorizer()), ("svc", svm.LinearSVC())])

    # set the range of parameters to be tuned
    parameters = {'tfidf__min_df': [1, 2, 5],
                  'tfidf__stop_words': [None, "english"],
                  'svc__C': [0.5, 1.5], }

    # the metric used to select the best parameters
    metric = "f1_macro"

    # gridSearch with cross validation
    gs_clf = GridSearchCV(text_clf, param_grid=parameters,
                          scoring=metric, cv=6)

    # find the best parameters
    gs_clf = gs_clf.fit(training_file["text"], training_file["label"])
    for param_name in gs_clf.best_params_:
        print(param_name, ": ", gs_clf.best_params_[param_name])
    print("best f1 score from GridSearch:", gs_clf.best_score_)

    # initialize the TfidfVectorizer with new parameters
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=1)

    # Generate tfidf matrix
    dtm = tfidf_vect.fit_transform(training_file["text"])
    dtm_test = tfidf_vect.transform(testing_file["text"])

    # initiate an linear SVM model with parameter C
    clf = svm.LinearSVC(C=0.5).fit(dtm, training_file["label"])

    # get the list of unique labels
    labels = sorted(training_file["label"].unique())

    # predicted testing data by SVM model
    predicted = clf.predict(dtm_test)

    # Create the classification report
    print("\nclassifcation_report:")
    print(classification_report(
        testing_file["label"], predicted, target_names=labels))


def impact_of_sample_size(train_file, test_file):

    # import training file and testing file
    train_file = pd.read_csv(train_file)
    test_file = pd.read_csv(test_file)
    i = 300
    x_train = train_file["text"].head(i)
    y_train = train_file["label"].head(i)
    x_test = test_file["text"]
    y_test = test_file["label"]

    # define the metrics
    metrics = ["precision_macro", "recall_macro"]

    # lists for multinomial Naive Bayes model
    macro_precision1 = []
    macro_recall1 = []

    # lists for linear support vector machine model
    macro_precision2 = []
    macro_recall2 = []

    # lists for samplesize of each iteration
    samplesize = []

    # get the list of unique labels
    labels = sorted(train_file["label"].unique())

    # For each iteration, sample size increases 300, two models' performance will
    # be recalculated
    while i <= len(train_file):
        # with stop words removed
        tfidf_vect1 = TfidfVectorizer(stop_words="english")
        # generate tfidf matrix
        dtm1 = tfidf_vect1.fit_transform(x_train)
        # train a classifier using multinomial Naive Bayes model
        clf1 = MultinomialNB().fit(dtm1, y_train)
        # train a classifier using linear support vector machine model
        clf2 = svm.LinearSVC().fit(dtm1, y_train)
        # generate tfidf matrix for testing
        dtm1_test = tfidf_vect1.transform(x_test)
        # predict the test data on multinomial Naive Bayes model
        predicted1 = clf1.predict(dtm1_test)
        # predict the test data on linear support vector machine model
        predicted2 = clf2.predict(dtm1_test)
        # Testing the performance of multinomial Naive Bayes model
        precision1, recall1, fscore1, support1 = precision_recall_fscore_support(
            y_test, predicted1, labels=labels, average="macro")
        # Testing the performance of linear support vector machine model
        precision2, recall2, fscore2, support2 = precision_recall_fscore_support(
            y_test, predicted2, labels=labels, average="macro")
        # For each iteration, precision and recall are added
        # into the list defined outside the loop
        macro_precision1.append(precision1)
        macro_recall1.append(recall1)

        macro_precision2.append(precision2)
        macro_recall2.append(recall2)
        
        samplesize.append(i)
        # For each iteration, sample size of training data increases 300
        i += 300
        x_train = train_file["text"].head(i)
        y_train = train_file["label"].head(i)

    # figure showing the relationship between sample size and precision
    plt.plot(samplesize, macro_precision1)
    plt.plot(samplesize, macro_precision2)
    plt.legend(["precision_NB", "precision_SVM"], loc="lower right")
    plt.xlabel("sample size")
    plt.ylabel("precision rate")
    plt.show()

    plt.plot(samplesize, macro_recall1)
    plt.plot(samplesize, macro_recall2)
    plt.legend(["recall_NB", "recall_SVM"], loc="lower right")
    plt.xlabel("sample size")
    plt.ylabel("recall rate")
    plt.show()


# import training file and testing file and call the classify function
training_file = r"C:\Users\TC\Desktop\BIA660\BIA660\Assignment5 - Classification\news_train.csv"
testing_file = r"C:\Users\TC\Desktop\BIA660\BIA660\Assignment5 - Classification\news_test.csv"
classify(training_file, testing_file)

# import train file and test file and call the impact_of_sample_size function
train_file = r"C:\Users\TC\Desktop\BIA660\BIA660\Assignment5 - Classification\news_train.csv"
test_file = r"C:\Users\TC\Desktop\BIA660\BIA660\Assignment5 - Classification\news_test.csv"
impact_of_sample_size(train_file, test_file)
