import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
       
class fastPlsa(object):

    """
    A collection of documents to perform PLSA on.
    """

    def __init__(self, documents_path_test = None, documents_path_train = None, labels_path_train = None, model_name = None):
        """
        Initialize empty document list.
        """
        ## initialization variables
        self.documents_train = []
        self.labels = []
        self.documents_test = []
        self.vocabulary = {}

        ## document paths
        self.documents_path_test = documents_path_test
        self.documents_path_train = documents_path_train
        self.labels_path_train = labels_path_train

        ## base features
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        ## background model extension
        self.background_word_prob = None # P(w | B)
        self.background_prob = None # P(B | d, w)
        self.mixing_weight = None # lambda_B

        ## prior extension
        self.prior_weights = None # mu
        self.prior_word_prob = None # P(w | z')

        ## aggregates
        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.number_of_topics = 0

        ## counters
        self.likelihoods = []
        self.em_iteration = 0

        if model_name:
            self.load_model(model_name)

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        self.documents_test = []
        f = open(self.documents_path_test, "r")
        for line in f:
            self.documents_test.append(line.split())
        self.number_of_documents = len(self.documents_test)

        if self.documents_path_train is not None:
            f = open(self.documents_path_train, "r")
            for line in f:
                self.documents_train.append(line.split())
            f = open(self.labels_path_train, "r")
            for line in f:
                self.labels.append(int(line))

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        temp_set = set.union(*map(set, self.documents_test))
        indices = range(len(temp_set))
        
        self.vocabulary = dict(zip(temp_set, indices))
        self.vocabulary_size = len(self.vocabulary)

    def build_prior(self, number_of_topics = 2):
        ## training files provided
        if len(self.documents_train) > 0:
            vocabulary_set = set(self.vocabulary.keys())
            self.number_of_topics = len(set(self.labels))
            self.prior_word_prob = np.zeros((self.number_of_topics, len(self.vocabulary)))
            self.prior_weights = np.zeros(self.number_of_topics)
            for i in range(len(self.documents_train)):
                for k, v in Counter(self.documents_train[i]).items():
                    if k not in vocabulary_set:
                        continue
                    self.prior_word_prob[self.labels[i]][self.vocabulary[k]] += v
                    self.prior_weights[self.labels[i]] += v

            self.prior_word_prob /= self.prior_word_prob.sum(axis = 1, keepdims = True)

        ## training files not provided
        else:
            self.number_of_topics = number_of_topics
            self.prior_weights = np.zeros((self.number_of_topics))
            self.prior_word_prob = np.zeros([self.number_of_topics, self.vocabulary_size])

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """

        self.term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))
        self.background_word_prob = np.zeros(self.vocabulary_size)
        for i in range(self.number_of_documents):
            for k, v in Counter(self.documents_test[i]).items():
                self.term_doc_matrix[i][self.vocabulary[k]] += v
                self.background_word_prob[self.vocabulary[k]] += v

        self.background_word_prob /= np.sum(self.background_word_prob)

    def initialize_randomly(self):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        self.document_topic_prob = np.random.rand(self.number_of_documents, self.number_of_topics)
        self.topic_word_prob = np.random.rand(self.number_of_topics, self.vocabulary_size)

    def initialize_uniformly(self):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, self.number_of_topics))
        self.document_topic_prob /= self.document_topic_prob.sum(axis = 1, keepdims = True)

        self.topic_word_prob = np.ones((self.number_of_topics, len(self.vocabulary)))
        self.topic_word_prob /= self.topic_word_prob.sum(axis = 1, keepdims = True)


    def initialize(self, random=False, mixing_weight = 0.3, number_of_topics = 2):
        """ Call the functions to initialize the fastPlsa object
        """
        print("Initializing...")

        print("Building corpus...")
        self.build_corpus()

        print("Building vocabulary...")
        self.build_vocabulary()

        print("Building priors...")
        self.build_prior()

        print("Building term-doc matrix...")
        self.build_term_doc_matrix()

        print("Initializing features...")
        if random:
            self.initialize_randomly()
        else:
            self.initialize_uniformly()

        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, self.number_of_topics, self.vocabulary_size])

        # P(B | d, w)
        self.background_prob = np.zeros([self.number_of_documents, self.vocabulary_size])

        # lambda_B
        self.mixing_weight = mixing_weight

        # reset iterations
        self.em_iteration = 0
        self.likelihoods = []

        print("Initialization complete")
        print("Number of topics: " + str(self.number_of_topics))
        print("Vocabulary size: " + str(len(self.vocabulary)))
        print("Number of documents: " + str(len(self.documents_test)))

    def expectation_step(self):
        """ The E-step updates P(z | w, d) and P(B | w, d)
        """
        print("E step:")
        # P(z | w, d) 
        self.topic_prob = np.einsum('dj,jw->djw', self.document_topic_prob, self.topic_word_prob)
        self.topic_prob /= self.topic_prob.sum(axis = 1, keepdims = True)

        # P(B | w, d)
        numerator = np.einsum(',w,d->dw', self.mixing_weight, self.background_word_prob, np.ones(self.number_of_documents))
        denominator = numerator + np.einsum(',dj,jw->dw',  (1 - self.mixing_weight), self.document_topic_prob, self.topic_word_prob)
        self.background_prob = numerator / denominator   

    def maximization_step(self):
        """ The M-step updates P(z | d) and P(w | z)
        """
        print("M step:")
        # P(z | d) 
        numerator = 1 + np.einsum('dw,dw,djw->dj', self.term_doc_matrix, (1 - self.background_prob), self.topic_prob)
        self.document_topic_prob = numerator / numerator.sum(axis = 1, keepdims = True)

        # P(w | z)
        numerator = np.einsum('j,jw->jw', self.prior_weights, self.prior_word_prob)
        numerator += np.einsum('dw,dw,djw->jw', self.term_doc_matrix, (1 - self.background_prob), self.topic_prob)
        self.topic_word_prob = numerator / numerator.sum(axis = 1, keepdims = True)

    def calculate_likelihood(self):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        sum_topic = np.einsum('dj,jw->dw', self.document_topic_prob, self.topic_word_prob)
        loglikelihood = np.einsum('dw,dw->', self.term_doc_matrix, np.log(sum_topic))
        
        print("likelihood: ", loglikelihood)
        self.likelihoods.append(loglikelihood)
        return

    def plsa(self, iters = 1, epsilon = 0.001, save_every_iter = 0, model_name = 'model.plsa'):

        """
        Model topics.
        """
        print ("EM iteration begins...")        

        for iteration in range(iters):
            print("Iteration #" + str(self.em_iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step()
            self.calculate_likelihood()
            self.em_iteration += 1

            ## autosave
            if save_every_iter > 0 and self.em_iteration % save_every_iter == 0:
                self.save_model(model_name = 'model')

            ## early stop
            if len(self.likelihoods) > 1:
                if np.abs(self.likelihoods[-1] - self.likelihoods[-2]) < epsilon:
                    print("Early stopped")
                    return

    def save_model(self, model_name = 'model'):
        print("Saving model...")
        np.savez_compressed(model_name,
            term_doc_matrix = self.term_doc_matrix, 
            document_topic_prob = self.document_topic_prob,
            topic_word_prob = self.topic_word_prob,
            topic_prob = self.topic_prob,
            background_word_prob = self.background_word_prob,
            background_prob = self.background_prob,
            mixing_weight = self.mixing_weight,
            prior_weights = self.prior_weights,
            prior_word_prob = self.prior_word_prob,
            likelihoods = np.array(self.likelihoods),
            counters = np.array([self.number_of_documents, self.vocabulary_size, self.number_of_topics, self.em_iteration]))

    def load_model(self, model_name = 'model'):
        print("Loading model...")
        loaded = np.load(model_name + '.npz')
        self.term_doc_matrix = loaded['term_doc_matrix']
        self.document_topic_prob = loaded['document_topic_prob']
        self.topic_word_prob = loaded['topic_word_prob']
        self.topic_prob = loaded['topic_prob']
        self.background_word_prob = loaded['background_word_prob']
        self.background_prob = loaded['background_prob']
        self.mixing_weight = loaded['mixing_weight']
        self.prior_weights = loaded['prior_weights']
        self.prior_word_prob = loaded['prior_word_prob']
        self.likelihoods = loaded['likelihoods'].tolist()

        self.number_of_documents = loaded['counters'][0]
        self.vocabulary_size = loaded['counters'][1]
        self.number_of_topics = loaded['counters'][2]
        self.em_iteration = loaded['counters'][3]

    def show_status(self):
        print("Number of topics:" + str(self.number_of_topics))
        print("Vocabulary size:" + str(len(self.vocabulary)))
        print("Number of documents:" + str(len(self.documents_test)))
        print("Current EM iterations: " + str(self.em_iteration))
        print("Current likelihoods: " + str(self.likelihoods))

    def evaluate_model(self, ground_truth_labels):
        print("Confusion matrix:")
        cm = confusion_matrix(ground_truth_labels, np.argmax(self.document_topic_prob, axis = 1))
        print(cm)
        print("Accuracy: ", sum(np.diag(cm)) / np.sum(cm))
