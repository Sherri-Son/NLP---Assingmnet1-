# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


# Problem (logistic_regression): Unigram features 
class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. 
    Each individual word is used as a bag-of-words feature
    Feature values are word-frequency counts
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for word in sentence:
            feat_name = "Unigram=" + word
            feat_idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] += 1.0
        return feats


# Problem (feature_exploration): BigramFeatureExtractor
class BigramFeatureExtractor(FeatureExtractor):
    """
    Indicator features for adjacent word pairs. Bigrams are indicator features, so each present bigram has value 1
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
 
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for i in range(len(sentence) - 1):
            feat_name = "Bigram=" + sentence[i] + "|" + sentence[i + 1]
            feat_idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if feat_idx != -1:
                # Bigram features are indicators, so the value is always 1.
                feats[feat_idx] = 1.0
        return feats


# Problem (feature_exploration): BetterFeatureExtractor

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better features using binary unigrams, bigrams, and trigrams.
    Binary/clipped unigram counts (a word contributes at most 1)
    Bigram indicators/ Trigram indicators as the additional feature modification beyond simply combining unigrams and bigrams
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()

        # use each unigram only once
        for word in set(sentence):
            feat_name = "Unigram=" + word
            feat_idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] = 1.0

        # bigrams
        for i in range(len(sentence) - 1):
            feat_name = "Bigram=" + sentence[i] + "|" + sentence[i + 1]
            feat_idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] = 1.0

        # trigrams are the extra feature modification
        for i in range(len(sentence) - 2):
            feat_name = "Trigram=" + sentence[i] + "|" + sentence[i + 1] + "|" + sentence[i + 2]
            feat_idx = self.indexer.add_and_get_index(feat_name, add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] = 1.0

        return feats


def _sigmoid(score):
    # Numerically stable sigmoid.
    if score >= 0:
        return 1.0 / (1.0 + np.exp(-score))
    exp_score = np.exp(score)
    return exp_score / (1.0 + exp_score)


# Problem (logistic_regression): Logistic-regression classifier

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Binary logistic regression classifier over sparse feature vectors.
    Computes a weighted sum of sparse features.
    Predicts positive (1) when score >= 0, otherwise negative (0)
    """ 

    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        feats = self.feat_extractor.extract_features(ex_words, False)
        score = 0.0
        for feat_idx, feat_val in feats.items():
            score += self.weights[feat_idx] * feat_val
        return 1 if score >= 0.0 else 0


# Problem (logistic_regression): Train unigram logistic regression
def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train binary logistic regression with SGD.
    (1) Builds the feature vocabulary
    (2) Uses stochastic gradient descent (SGD)
    (3) Default unigram path is selected in train_linear_model below
    """
    random.seed(0)

    # First pass: build the vocabulary and cache sparse feature vectors.
    train_cache = []
    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, True)
        train_cache.append((feats, ex.label))

    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    num_epochs = 30
    learning_rate = 0.1

    # SGD: maximize log p(y | x).
    for epoch in range(num_epochs):
        random.shuffle(train_cache)

        for feats, label in train_cache:
            score = 0.0
            for feat_idx, feat_val in feats.items():
                score += weights[feat_idx] * feat_val

            prob_pos = _sigmoid(score)
            error = label - prob_pos

            for feat_idx, feat_val in feats.items():
                weights[feat_idx] += learning_rate * error * feat_val

    return LogisticRegressionClassifier(weights, feat_extractor)


# Problem (lr_schedule_exploration): Step-size schedules
# (1) Compares constant step sizes, decay by epoch, and 1/t decay
# (2) Records training accuracy, development accuracy, and log likelihood
# (3) run_lr_experiment() creates the required matplotlib plots

def get_score(weights, feats):
    score = 0.0
    for idx, value in feats.items():
        score += weights[idx] * value
    return score


def get_accuracy(weights, data):
    correct = 0
    for feats, label in data:
        score = get_score(weights, feats)
        pred = 1 if score >= 0 else 0
        if pred == label:
            correct += 1
    return correct / len(data)


def get_log_likelihood(weights, data):
    total = 0.0
    for feats, label in data:
        score = get_score(weights, feats)
        p = _sigmoid(score)
        # avoid log(0)
        p = min(max(p, 1e-10), 1 - 1e-10)
        total += label * np.log(p) + (1 - label) * np.log(1 - p)
    return total


# Problem (lr_schedule_exploration): run all schedules and plot results
def run_lr_experiment(train_exs, dev_exs, num_epochs=20):
    import matplotlib.pyplot as plt

    # constant schedule with two step sizes, plus the two schedules from the prompt
    settings = [
        ("constant 0.1", "constant", 0.1),
        ("constant 0.3", "constant", 0.3),
        ("decay by epoch", "decay", 0.3),
        ("1/t decay", "1/t", 0.3)
    ]

    all_results = {}

    for name, schedule, start_lr in settings:
        random.seed(0)
        feat_extractor = UnigramFeatureExtractor(Indexer())

        train_data = []
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, True)
            train_data.append((feats, ex.label))

        dev_data = []
        for ex in dev_exs:
            feats = feat_extractor.extract_features(ex.words, False)
            dev_data.append((feats, ex.label))

        weights = np.zeros(len(feat_extractor.get_indexer()))
        log_likelihoods = []
        train_accs = []
        dev_accs = []
        t = 0

        for epoch in range(num_epochs):
            random.shuffle(train_data)

            for feats, label in train_data:
                t += 1

                if schedule == "constant":
                    lr = start_lr
                elif schedule == "decay":
                    lr = start_lr * (0.9 ** epoch)
                else:
                    lr = start_lr / (1 + 0.001 * t)

                score = get_score(weights, feats)
                p = _sigmoid(score)
                error = label - p

                for idx, value in feats.items():
                    weights[idx] += lr * error * value

            log_likelihoods.append(get_log_likelihood(weights, train_data))
            train_accs.append(get_accuracy(weights, train_data))
            dev_accs.append(get_accuracy(weights, dev_data))

        all_results[name] = (log_likelihoods, train_accs, dev_accs)
        print(name, "train accuracy:", train_accs[-1], "dev accuracy:", dev_accs[-1])

    epochs = range(1, num_epochs + 1)

    plt.figure()
    for name, result in all_results.items():
        plt.plot(epochs, result[0], label=name)
    plt.xlabel("Training iteration (epoch)")
    plt.ylabel("Dataset log likelihood")
    plt.title("Training Log Likelihood")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lr_log_likelihood.png")
    plt.show()

    plt.figure()
    for name, result in all_results.items():
        plt.plot(epochs, result[2], label=name)
    plt.xlabel("Training iteration (epoch)")
    plt.ylabel("Development Accuracy")
    plt.title("Development Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lr_dev_accuracy.png")
    plt.show()

    return all_results


# Problems (logistic_regression + feature_exploration)
def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


# Problem (dan): Implement the Deep Averaging Network
class DeepAveragingNetwork(nn.Module):
    """
    Deep Averaging Network:
    embeddings -> masked average -> hidden layer -> ReLU -> dropout -> output
    (1) Look up pretrained word embeddings
    (2) Average the word vectors for each sentence
    (3) Feed the average through a feedforward neural network
    """
    def __init__(self, word_embeddings, hidden_size=100, dropout=0.3):
        super().__init__()

        self.embedding = word_embeddings.get_initialized_embedding_layer(
        frozen=False)


        embedding_dim = word_embeddings.get_embedding_length()

        
        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, word_ids):
        # word_ids: [batch_size, sentence_length]
        embedded = self.embedding(word_ids)  # [B, L, D]

        # Ignore PAD tokens when averaging.
        mask = (word_ids != 0).float().unsqueeze(-1)  # [B, L, 1]
        summed = (embedded * mask).sum(dim=1)         # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1.0)      # [B, 1]
        averaged = summed / lengths

        hidden = self.relu(self.hidden(averaged))
        hidden = self.dropout(hidden)
        return self.output(hidden)                    # [B, 2]



# Problem (batching_exploration): Build padded training batches
def _make_batch(examples, word_embeddings, max_len=60, min_len=1):
    """
    Pads a batch of SentimentExample objects with PAD=0.
    Sentences have different lengths, so pad them with PAD=0
    This lets the network process multiple examples at once
    max_len=60 keeps training fast; longer sentences are truncated.
    """
    encoded = [_words_to_ids(ex.words, word_embeddings)[:max_len] for ex in examples]
    batch_len = max(min_len, max(len(ids) for ids in encoded))

    x = torch.zeros((len(examples), batch_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        if len(ids) > 0:
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    y = torch.tensor([ex.label for ex in examples], dtype=torch.long)
    return x, y


# Problem (batching_exploration): padded batches for prediction/test time
def _make_prediction_batch(sentences, word_embeddings, max_len=60, min_len=1):
    """Same padding logic as _make_batch, but for unlabeled sentences."""
    encoded = [_words_to_ids(words, word_embeddings)[:max_len] for words in sentences]
    batch_len = max(min_len, max(len(ids) for ids in encoded))

    x = torch.zeros((len(sentences), batch_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        if len(ids) > 0:
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x



# Problems (dan + batching_exploration)
class NeuralSentimentClassifier(SentimentClassifier):
    """Wraps a trained PyTorch network for prediction."""
    def __init__(self, network, word_embeddings, min_len=1):
        self.network = network
        self.word_embeddings = word_embeddings
        self.min_len = min_len

    def predict(self, ex_words: List[str]) -> int:
        self.network.eval()
        with torch.no_grad():
            x = _make_prediction_batch(
                [ex_words], self.word_embeddings, min_len=self.min_len
            )
            logits = self.network(x)
            return int(torch.argmax(logits, dim=1).item())

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        self.network.eval()
        predictions = []
        test_batch_size = 128

        with torch.no_grad():
            for start in range(0, len(all_ex_words), test_batch_size):
                batch_words = all_ex_words[start:start + test_batch_size]
                x = _make_prediction_batch(
                    batch_words, self.word_embeddings, min_len=self.min_len
                )
                logits = self.network(x)
                predictions.extend(torch.argmax(logits, dim=1).tolist())

        return predictions


def train_deep_averaging_network(args, train_exs: List[SentimentExample],
                                 dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    
    torch.manual_seed(0)
    random.seed(0)

    network = DeepAveragingNetwork(
        word_embeddings,
        hidden_size=args.hidden_size,
        dropout=0.3
    )
    # CrossEntropyLoss + Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    # Problem (batching_exploration)
    # Use the command-line batch size so loss is computed over
    # a whole batch instead of one example at a time.
    batch_size = 32

    for epoch in range(args.num_epochs):
        network.train()
        random.shuffle(train_exs)
        total_loss = 0.0

        for start in range(0, len(train_exs), batch_size):
            batch = train_exs[start:start + batch_size]
            x, y = _make_batch(
                batch,
                word_embeddings,
                max_len=60,
                min_len=min_len
            )

            optimizer.zero_grad()
            logits = network(x)

            # Problem (batching_exploration): y contains all labels
            # in this batch, so this computes loss over the full batch.
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(train_exs)
        print("Epoch %d loss: %.4f" % (epoch + 1, avg_loss))

    return NeuralSentimentClassifier(network, word_embeddings, min_len=min_len)
