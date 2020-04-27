# import nltk
# from nltk.stem import WordNetLemmatizer
import regex as re
import random
# lemmatizer = WordNetLemmatizer()


class Classifier:

    def __init__(self, file):
        self.raw_data_file = file
        self.convert_txt_to_rows(file)
        random.shuffle(self.data)
        n = len(self.data) // 5
        folds = [self.data[x:x + n] for x in range(0, len(self.data), n)]
        train_data = []
        test_data = []

        for i in range(5):  # 5 fold cross validation
            train_data = []
            test_data = folds[i]
            for j in range(1, 5):
                train_data += folds[(i + j) % 5]
            self.train(train_data)
            self.test(test_data)

    def convert_txt_to_rows(self, file):
        f = open(file)
        self.data = []

        for line in f.readlines():
            line = line.replace('\t', " ")
            line = line.replace("\n", "")
            line = line.replace(".", "")
            # line = re.sub (r'([^a-zA-Z ])', '', line)
            line = line.lower()
            row = line.split(" ")
            row = [word for word in row if (word != "")]
            for i in range(len(row) - 1):
                # row[i] = lemmatizer.lemmatize(row[i])
                row[i] = re.sub(r'([^a-zA-Z ])', '', row[i])
            self.data.append(row)

    def train(self, train_data):
        # 0. Splitting
        # 1. lammetization // ignored since nltk cannot be used
        # 2. Count each word's frequency and sentiment

        train_len = (int)(len(train_data))
        data = train_data
        training = dict()
        total_words_pos_sent = 0
        total_words_neg_sent = 0
        tot_pos = 0
        tot_neg = 0
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
            for j in range(len(data[i]) - 1):
                word = data[i][j]
                if(word == ""):
                    continue
                if(word not in training):
                    training[word] = dict()
                if(sentiment not in training[word]):
                    training[word][sentiment] = 0
                training[word][sentiment] += 1

        self.training = training
        self.total_words_neg_sent = total_words_neg_sent
        self.total_words_pos_sent = total_words_pos_sent
        self.tot_pos = tot_pos
        self.tot_neg = tot_neg

    def test(self, test_data):
        correctly_classified = 0
        tot_dist_words = len(self.training.keys())
        for i in range(len(test_data)):
            data = test_data
            sentiment_ans = data[i][len(data[i]) - 1]
            res0 = 1
            res1 = 1
            for j in range(len(data[i]) - 1):
                word = data[i][j]
                if(word not in self.training):
                    self.training[word] = dict()
                if(0 not in self.training[word]):
                    self.training[word][0] = 0
                if(1 not in self.training[word]):
                    self.training[word][1] = 0
                res0 *= (self.training[word][0] + 1) / (self.total_words_neg_sent +
                                                        tot_dist_words)  # laplacesmoothing
                res1 *= (self.training[word][1] + 1) / \
                    (self.total_words_pos_sent + tot_dist_words)

            res0 *= self.tot_neg
            res1 *= self.tot_pos

            if(res0 > res1):

                if(int(sentiment_ans) == 0):
                    correctly_classified += 1
                    continue
            elif(res1 > res0):
                if(int(sentiment_ans) == 1):
                    correctly_classified += 1
                    continue

        print(correctly_classified / len(test_data))


classifier = Classifier("./a1_d3.txt")
