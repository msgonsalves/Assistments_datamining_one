import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torchvision.transforms as transforms
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from scipy import interp
from sklearn import metrics
from sklearn import tree

import torchvision

from csv_parser_helper import *

NUM_PROBLEMS = 24
ALL_PROBLEMS_FILE = "alt_time_for_problems_file"
FIRST_TEN_MINUTES_FILE = "ten_minutes"
FIRST_TWENTY_MINUTES_FILE = "twenty_minutes"
TRAINING_FILE = "training_labels"
PROB_ID_FILES = "prob_ids"


def get_files():
    print("hey")
    the_data = []
    file = open(ALL_PROBLEMS_FILE, 'rb')
    all_problems = pickle.load(file)
    the_data.append(all_problems)
    file.close()
    file = open(FIRST_TEN_MINUTES_FILE, 'rb')
    ten_min_data = pickle.load(file)
    the_data.append(ten_min_data)
    file.close()
    file = open(FIRST_TWENTY_MINUTES_FILE, 'rb')
    twenty_min_data = pickle.load(file)
    the_data.append(twenty_min_data)
    file.close()
    file = open(TRAINING_FILE, 'rb')
    training_labels = pickle.load(file)
    the_data.append(training_labels)
    file.close()
    file = open(PROB_ID_FILES, 'rb')
    prob_ids = pickle.load(file)
    the_data.append(prob_ids)
    file.close()

    return the_data


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.input_layer = nn.Linear(NUM_PROBLEMS * 7, 50)
        self.layer_one = nn.Linear(50, 50)
        self.layer_two = nn.Linear(50, 50)
        self.layer_three = nn.Linear(50, 50)
        self.layer_four = nn.Linear(50, 50)
        self.layer_five = nn.Linear(50, 50)
        self.layer_six = nn.Linear(50, 25)
        self.layer_seven = nn.Linear(25, 1)

    def forward(self, input):
        input = F.sigmoid(self.input_layer(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_one(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_two(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_three(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_four(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_five(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_six(input))

        # input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_seven(input))

        return input


def my_loss(label, output):
    if label > output:
        return (label - output)
    else:
        return .75 * (output - label)


def train_network(neural_net, optimizer, training_data, training_labels, criterion):
    for epoch in range(0, 4):

        running_loss = 0
        for i in range(0, 900):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = neural_net(training_data[i])

            loss = my_loss(outputs, training_labels[i:i + 1].float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    return neural_net


def make_training_data(a_list, prob_ids, is_tensor):
    map = a_list[1]

    input_list = []
    for a_prob in prob_ids:
        if a_prob in map:
            input_list.append(map[a_prob][0])
            input_list.append(map[a_prob][1])
            input_list.append(map[a_prob][2])
            input_list.append(map[a_prob][3])
            input_list.append(map[a_prob][4])
            input_list.append(map[a_prob][5])
            input_list.append(map[a_prob][6])
        else:
            pass
            # input_list.append(-1)
            # input_list.append(-5)
            # input_list.append(-5)
            # input_list.append(-5)
            # input_list.append(-5)
    # input_list.append(a_list[2])
    # input_list.append(a_list[3])
    # input_list.append(a_list[4])
    if is_tensor:
        return torch.FloatTensor(input_list)
    else:
        return input_list


def run_cross(all_data, ten_data, twenty_data, training_labels, prob_ids, num_splits):
    labels = []
    for a_label in training_labels:
        labels.append(a_label[1])
    labels = np.array(labels)

    ten_features = []
    twenty_features = []
    tot_features = []
    for a_person in all_data:
        tot_features.append(make_training_data(a_person, prob_ids, False))
    tot_features = np.array(tot_features)

    for a_person in ten_data:
        ten_features.append(make_training_data(a_person, prob_ids, False))
    ten_features = np.array(ten_features)

    for a_person in twenty_data:
        twenty_features.append(make_training_data(a_person, prob_ids, False))
    twenty_features = np.array(twenty_features)

    cv = StratifiedKFold(n_splits=num_splits)

    ten_rf = RandomForestRegressor(n_estimators=50)
    twenty_rf = RandomForestRegressor(n_estimators=50)
    tot_rf = RandomForestRegressor(n_estimators=50)
    dt = tree.DecisionTreeRegressor()
    bc = BaggingRegressor(n_estimators=50)
    etr = ExtraTreesRegressor(n_estimators=50)

    ten_aucs = []
    twenty_aucs = []
    tot_aucs = []

    for train, test in cv.split(ten_features, labels):
        ten_rf.fit(ten_features[train], labels[train])
        predictions = ten_rf.predict(ten_features[test])

        fpr, tpr, thresholds = metrics.roc_curve(labels[test], predictions)

        roc_auc = metrics.auc(fpr, tpr)
        ten_aucs.append(roc_auc)
        # print(roc_auc)

    for train, test in cv.split(twenty_features, labels):
        twenty_rf.fit(twenty_features[train], labels[train])
        predictions = twenty_rf.predict(twenty_features[test])

        fpr, tpr, thresholds = metrics.roc_curve(labels[test], predictions)

        roc_auc = metrics.auc(fpr, tpr)
        twenty_aucs.append(roc_auc)
        # print(roc_auc)

    for train, test in cv.split(tot_features, labels):
        tot_rf.fit(tot_features[train], labels[train])
        predictions = tot_rf.predict(tot_features[test])

        fpr, tpr, thresholds = metrics.roc_curve(labels[test], predictions)

        roc_auc = metrics.auc(fpr, tpr)
        tot_aucs.append(roc_auc)
        # print(roc_auc)
    selector = RFE(tot_rf)
    selector = selector.fit(tot_features, labels)

    print(selector.ranking_)
    print(selector.support_)
    print("avg ten auc with ", num_splits, " folds = ", round(sum(ten_aucs) / len(ten_aucs), 3))
    print("avg twenty auc with ", num_splits, " folds = ", round(sum(twenty_aucs) / len(twenty_aucs), 3))
    print("avg tot auc with ", num_splits, " folds = ", round(sum(tot_aucs) / len(tot_aucs), 3))

    return (ten_rf, twenty_rf, tot_rf)


def run_random(all_data, ten_data, twenty_data, training_labels, prob_ids):
    labels = []
    for a_label in training_labels:
        labels.append(a_label[1])
    labels = np.array(labels)

    features = []
    for a_person in all_data:
        features.append(make_training_data(a_person, prob_ids, False))
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=45)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    rf = RandomForestRegressor(n_estimators=500, random_state=45)
    bc = BaggingRegressor(n_estimators=1000)
    # bc.fit(train_features, train_labels)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    # predictions = bc.predict(test_features)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions)
    print("auc: ", metrics.auc(fpr, tpr))

    all_round_point = []
    start_point = 0.1
    for i in range(0, 360):
        all_round_point.append(start_point)
        start_point += .0025

    accuracies = []
    for a_cut in all_round_point:
        correct = 0
        other_tot = 0
        total_wrong = 0
        num = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for i in range(0, len(test_labels)):
            output = predictions[i]
            if output > a_cut:
                temp_other = 1
            else:
                temp_other = 0
            if not temp_other == test_labels[i]:
                if temp_other == 1:
                    false_positive += 1
                else:
                    false_negative += 1
                other_tot += 1
            else:
                if temp_other == 1:
                    true_positive += 1
                else:
                    true_negative += 1
                correct += 1
            total_wrong += abs(output.item() - test_labels[i])
            num += 1
        accuracies.append(correct / num)
        if correct / num > .74:
            print("true_positive: ", true_positive)
            print("false_positive: ", false_positive)
            print("false_negative: ", false_negative)
            print("true_negative: ", true_negative)
            print("accuracy: ", correct / num, "cut point: ", a_cut)
        # print("roundin point: ", a_cut, " accuracy: ", correct/num, " number wrong: ", other_tot, " tot_off: ", total_wrong)

    plt.plot(all_round_point, accuracies, 'b-o', label='Accuracy')
    plt.xlabel("rounding point")
    plt.ylabel("accuracy")
    plt.show()


def main():
    all_data, ten_data, twenty_data, training_labels, prob_ids = get_files()

    train_ten_set = ten_data[0:900]
    train_twenty_set = twenty_data[0:900]
    train_all_set = all_data[0:900]
    train_labels = training_labels[0:900]
    test_ten_set = ten_data[900:1231]
    test_twenty_set = twenty_data[900:1231]
    test_all_set = all_data[900:1231]
    test_labels = training_labels[900:1231]

    # run_random(all_data, ten_data, twenty_data, training_labels, prob_ids)
    for i in range(5, 12, 5):
        ten_mod, twenty_mod, tot_mod = run_cross(train_all_set, train_ten_set, train_twenty_set, train_labels, prob_ids,
                                                 i)

    j_test_labels = []
    for a_label in test_labels:
        j_test_labels.append(a_label[1])

    ten_features = []
    twenty_features = []
    tot_features = []
    for a_person in test_all_set:
        tot_features.append(make_training_data(a_person, prob_ids, False))
    tot_features = np.array(tot_features)

    for a_person in test_twenty_set:
        ten_features.append(make_training_data(a_person, prob_ids, False))
    ten_features = np.array(ten_features)

    for a_person in test_ten_set:
        twenty_features.append(make_training_data(a_person, prob_ids, False))
    twenty_features = np.array(twenty_features)


    ten_predictions = ten_mod.predict(ten_features)
    fpr, tpr, thresholds = metrics.roc_curve(j_test_labels, ten_predictions)
    ten_roc_auc = metrics.auc(fpr, tpr)

    twenty_predictions = twenty_mod.predict(twenty_features)
    fpr, tpr, thresholds = metrics.roc_curve(j_test_labels, twenty_predictions)
    twenty_roc_auc = metrics.auc(fpr, tpr)

    tot_predictions = tot_mod.predict(tot_features)
    fpr, tpr, thresholds = metrics.roc_curve(j_test_labels, tot_predictions)
    tot_roc_auc = metrics.auc(fpr, tpr)

    file = open('ten_twenty_full_predictions.csv', 'w+', newline='')
    writer = csv.writer(file)
    writer.writerow(["ten minute prediction", "twenty minute prediction", "full prediction", "actual label", "student id"])
    for i in range(0,len(ten_predictions)):
        writer.writerow([ten_predictions[i], twenty_predictions[i], tot_predictions[i], j_test_labels[i], test_labels[i][0]])

    file.close()
    print("ten auc:", round(ten_roc_auc, 3))
    print("twenty auc:", round(twenty_roc_auc, 3))
    print("tot auc:", round(tot_roc_auc, 3))

    # print(ten_data[0])
    # print(twenty_data[0])
    # print(all_data[0])
    # print(all_data[1])
    # print(all_data[2])
    #
    #
    # just_labels = []
    # for a_label in training_labels:
    #     just_labels.append(a_label[1])
    #

    #
    # print(training_labels[0])
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # ten_min_training_set = []
    # twenty_min_training_set = []
    # total_min_training_set = []
    #
    # ten_min_test_set = []
    # twenty_min_test_set = []
    # total_min_test_set = []
    # #
    #
    # for i in range(0, len(train_ten_set)):
    #     temp_ten_tensor = make_training_data(train_ten_set[i], prob_ids, True)
    #     ten_min_training_set.append(temp_ten_tensor)
    #
    #     temp_twenty_tensor = make_training_data(train_twenty_set[i], prob_ids, True)
    #     twenty_min_training_set.append(temp_twenty_tensor)
    #
    #     temp_total_tensor = make_training_data(train_all_set[i], prob_ids, True)
    #     total_min_training_set.append(temp_total_tensor)
    #
    # train_labels = torch.Tensor(train_labels)
    #
    # for i in range(0, len(test_ten_set)):
    #     temp_ten_tensor = make_training_data(test_ten_set[i], prob_ids)
    #     ten_min_test_set.append(temp_ten_tensor)
    #
    #     temp_twenty_tensor = make_training_data(test_twenty_set[i], prob_ids)
    #     twenty_min_test_set.append(temp_twenty_tensor)
    #
    #     temp_total_tensor = make_training_data(test_all_set[i], prob_ids)
    #     total_min_test_set.append(temp_total_tensor)
    #
    # # ten_neural_network = NeuralNet()
    # criterion = nn.L1Loss()
    # # ten_optimizer = optim.Adam(ten_neural_network.parameters(), lr=0.001)
    # #
    # # ten_neural_network = train_network(ten_neural_network, ten_optimizer, ten_min_training_set, train_labels, criterion)
    # #
    # # twenty_neural_network = NeuralNet()
    # # twenty_optimizer = optim.SGD(twenty_neural_network.parameters(), lr=0.001, momentum=0.5)
    # # twenty_neural_network = train_network(twenty_neural_network, twenty_optimizer, twenty_min_training_set,
    # #                                       train_labels, criterion)
    #
    # all_neural_network = NeuralNet()
    # all_optimizer = optim.SGD(all_neural_network.parameters(), lr=0.001, momentum=0.5)
    # all_neural_network = train_network(all_neural_network, all_optimizer, total_min_training_set, train_labels,
    #                                    criterion)
    #
    # print('Finished Training')
    # #
    # # total_wrong = 0
    # # other_tot = 0
    # # for i in range(0, len(test_labels)):
    # #     output = ten_neural_network(ten_min_test_set[i])
    # #     temp_other = round(output.item(), 0)
    # #
    # #     if not temp_other == test_labels[i]:
    # #         other_tot+=1
    # #     total_wrong += abs(output.item() - test_labels[i])
    # #     # print(total_wrong)
    # #
    # # print(other_tot)
    # # print(total_wrong)
    # #
    # # total_wrong = 0
    # # other_tot = 0
    # # for i in range(0, len(test_labels)):
    # #     output = twenty_neural_network(twenty_min_test_set[i])
    # #     temp_other = round(output.item(), 0)
    # #     if not temp_other == test_labels[i]:
    # #         other_tot += 1
    # #     total_wrong += abs(output.item() - test_labels[i])
    # #
    # # print(other_tot)
    # # print(total_wrong)
    #
    # total_wrong = 0
    # other_tot = 0
    # outputs = []
    # for i in range(0, len(test_labels)):
    #     output = all_neural_network(total_min_test_set[i])
    #     temp_other = round(output.item(), 0)
    #     if not temp_other == test_labels[i]:
    #         other_tot += 1
    #     outputs.append(output)
    #     print(output, temp_other, test_labels[i])
    #     total_wrong += abs(output.item() - test_labels[i])
    # print(other_tot)
    # print(total_wrong)
    #
    # all_round_point = []
    # start_point = .45
    # for i in range(0, 200):
    #     all_round_point.append(start_point)
    #     start_point += .0025
    #
    # accuracies = []
    # for a_cut in all_round_point:
    #     correct = 0
    #     other_tot = 0
    #     total_wrong = 0
    #     num = 0
    #
    #     true_positive = 0
    #     false_positive = 0
    #     false_negative = 0
    #     true_negative = 0
    #     for i in range(0, len(test_labels)):
    #         output = outputs[i]
    #         if output > a_cut:
    #             temp_other = 1
    #         else:
    #             temp_other = 0
    #         if not temp_other == test_labels[i]:
    #             if temp_other == 1:
    #                 false_positive += 1
    #             else:
    #                 false_negative += 1
    #             other_tot += 1
    #         else:
    #             if temp_other == 1:
    #                 true_positive += 1
    #             else:
    #                 true_negative += 1
    #             correct += 1
    #         total_wrong += abs(output.item() - test_labels[i])
    #         num += 1
    #     accuracies.append(correct/num)
    #     if other_tot<130:
    #         print("true_positive: ", true_positive)
    #         print("false_positive: ", false_positive)
    #         print("false_negative: ", false_negative)
    #         print("true_negative: ", true_negative)
    #     print("roundin point: ", a_cut, " accuracy: ", correct/num, " number wrong: ", other_tot, " tot_off: ", total_wrong)
    #
    # plt.plot(all_round_point, accuracies, 'b-o', label='Accuracy')
    # plt.xlabel("rounding point")
    # plt.ylabel("accuracy")
    # plt.show()


if __name__ == "__main__":
    main()
