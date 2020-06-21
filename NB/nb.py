import regex as re
import random


class Classifier:

    def __init__(self, file):
        kfold = 5
        self.accuracy = 0
        self.raw_data_file = file
        self._convert_txt_to_rows(file)
        random.shuffle(self.data)
        n = len(self.data) // kfold
        folds = [self.data[x:x + n] for x in range(0, len(self.data), n)]

        # k fold cross validation
        for i in range(kfold):
            train_data = []
            test_data = folds[i]
            for j in range(1, kfold):
                train_data += folds[(i + j) % kfold]
            self._train(train_data)
            self._test(test_data)
        self.accuracy = round(self.accuracy / kfold, 2)
        print("overall avg accuracy: {}".format(self.accuracy))

    def _convert_txt_to_rows(self, file):
        f = open(file)
        self.data = []
        for line in f.readlines():
            line = line.replace("\t", " ")
            line = line.replace("\n", "")
            line = line.replace(".", "")
            line = line.lower()
            row = line.split(" ")
            row = [word for word in row if (word != "")]
            for i in range(len(row) - 1):
                row[i] = re.sub(r'([^a-zA-Z ])', '', row[i])
            self.data.append(row)

    def _train(self, train_data):
        # 1. Splitting
        # 2. Count each word's frequency and sentiment
        training = dict()
        train_len = (int)(len(train_data))
        data = train_data
        tot_pos = tot_neg = 0
        total_words_pos_sent = total_words_neg_sent = 0

        for i in range(train_len):
            sentiment = int(data[i][len(data[i]) - 1])  # last character
            if(sentiment not in [0, 1]):
                print("Error: ", data[i])
                print(sentiment)
            if(sentiment == 0):
                total_words_neg_sent += len(data[i]) - 1
                tot_neg += 1
            else:
                total_words_pos_sent += len(data[i]) - 1
                tot_pos += 1

            # find the positive and negative sentiments associated with each
            # word
            for j in range(len(data[i]) - 1):
                word = data[i][j]
                if(word == ""):
                    continue
                if(word not in training):
                    training[word] = dict()
                    training[word][0] = training[word][1] = 0
                training[word][sentiment] += 1

        # total number of postive and negative sentiments seen: P(C_k)
        self.tot_pos = tot_pos
        self.tot_neg = tot_neg
        self.training = training
        self.total_words_neg_sent = total_words_neg_sent
        self.total_words_pos_sent = total_words_pos_sent

    def _test(self, test_data):
        correctly_classified = 0
        tot_dist_words = len(self.training.keys())
        for i in range(len(test_data)):
            res0 = res1 = 1
            data = test_data
            sentiment_ans = data[i][len(data[i]) - 1]
            for j in range(len(data[i]) - 1):
                word = data[i][j]
                if(word not in self.training):
                    self.training[word] = dict()
                    self.training[word][0] = self.training[word][1] = 0
                # laplace_smoothing to avoid 0 probability case
                # P(C_k | x1, x2, x3 ... xn) = P(C_k) * P(x1 | C_k) * P(x2 |
                # C_k) ... P(xn | C_k)
                res0 *= (self.training[word][0] + 1) / \
                    (self.total_words_neg_sent + tot_dist_words)
                res1 *= (self.training[word][1] + 1) / \
                    (self.total_words_pos_sent + tot_dist_words)
            res0 *= self.tot_neg
            res1 *= self.tot_pos
            if((res0 > res1) and (int(sentiment_ans) == 0)) or ((res0 <= res1) and (int(sentiment_ans) == 1)):
                correctly_classified += 1

        current_accuracy = round(correctly_classified / len(test_data), 2)
        self.accuracy += current_accuracy
        print(current_accuracy)


classifier = Classifier("./a1_d3.txt")
