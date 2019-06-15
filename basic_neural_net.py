import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torchvision.transforms as transforms
import torch.optim as optim
import math

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

        self.input_layer = nn.Linear(NUM_PROBLEMS * 2, 100)
        self.layer_one = nn.Linear(100, 200)
        self.layer_two = nn.Linear(200, 200)
        self.layer_three = nn.Linear(200, 200)
        self.layer_four = nn.Linear(200, 100)
        self.layer_five = nn.Linear(100, 50)
        self.layer_six = nn.Linear(50, 25)
        self.layer_seven = nn.Linear(25, 1)

    def forward(self, input):
        input = F.relu(self.input_layer(input))
        input = F.dropout(input, training=self.training)
        input = F.relu(self.layer_one(input))
        input = F.dropout(input, training=self.training)
        input = F.relu(self.layer_two(input))
        input = F.dropout(input, training=self.training)
        input = F.relu(self.layer_three(input))
        input = F.dropout(input, training=self.training)
        input = F.relu(self.layer_four(input))
        input = F.dropout(input, training=self.training)
        input = F.relu(self.layer_five(input))
        input = F.dropout(input, training=self.training)
        input = F.relu(self.layer_six(input))
        input = F.dropout(input, training=self.training)
        input = F.sigmoid(self.layer_seven(input))

        return input


def train_network(neural_net, optimizer, training_data, training_labels, criterion):
    for epoch in range(0, 40):

        running_loss = 0
        for i in range(0, 900):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = neural_net(training_data[i])

            loss = criterion(outputs, training_labels[i:i + 1].float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    return neural_net


def make_training_data(a_list, prob_ids):
    map = a_list[1]

    input_list = []
    for a_prob in prob_ids:
        if a_prob in map:
            input_list.append(map[a_prob][0])
            input_list.append(map[a_prob][1])
        else:
            input_list.append(0)
            input_list.append(0)

    return torch.FloatTensor(input_list)


def main():
    all_data, ten_data, twenty_data, training_labels, prob_ids = get_files()

    print(ten_data[0])
    print(twenty_data[0])
    print(all_data[0])
    print(all_data[1])
    print(all_data[2])
    print(prob_ids)

    just_labels = []
    for a_label in training_labels:
        just_labels.append(a_label[1])

    train_ten_set = ten_data[0:900]
    train_twenty_set = twenty_data[0:900]
    train_all_set = all_data[0:900]
    train_labels = just_labels[0:900]
    test_ten_set = ten_data[900:1233]
    test_twenty_set = twenty_data[900:1233]
    test_all_set = all_data[900:1233]
    test_labels = just_labels[900:1233]

    print(training_labels[0])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ten_min_training_set = []
    twenty_min_training_set = []
    total_min_training_set = []

    ten_min_test_set = []
    twenty_min_test_set = []
    total_min_test_set = []
    #

    for i in range(0, len(train_ten_set)):
        temp_ten_tensor = make_training_data(train_ten_set[i], prob_ids)
        ten_min_training_set.append(temp_ten_tensor)

        temp_twenty_tensor = make_training_data(train_twenty_set[i], prob_ids)
        twenty_min_training_set.append(temp_twenty_tensor)

        temp_total_tensor = make_training_data(train_all_set[i], prob_ids)
        total_min_training_set.append(temp_total_tensor)

    train_labels = torch.Tensor(train_labels)

    for i in range(0, len(test_ten_set)):
        temp_ten_tensor = make_training_data(test_ten_set[i], prob_ids)
        ten_min_test_set.append(temp_ten_tensor)

        temp_twenty_tensor = make_training_data(test_twenty_set[i], prob_ids)
        twenty_min_test_set.append(temp_twenty_tensor)

        temp_total_tensor = make_training_data(test_all_set[i], prob_ids)
        total_min_test_set.append(temp_total_tensor)

    ten_neural_network = NeuralNet()
    criterion = nn.L1Loss()
    ten_optimizer = optim.SGD(ten_neural_network.parameters(), lr=0.001, momentum=0.05)

    ten_neural_network = train_network(ten_neural_network, ten_optimizer, ten_min_training_set, train_labels, criterion)

    twenty_neural_network = NeuralNet()
    twenty_optimizer = optim.SGD(twenty_neural_network.parameters(), lr=0.001, momentum=0.4)
    twenty_neural_network = train_network(twenty_neural_network, twenty_optimizer, twenty_min_training_set,
                                          train_labels, criterion)

    all_neural_network = NeuralNet()
    all_optimizer = optim.SGD(all_neural_network.parameters(), lr=0.001, momentum=0.4)
    all_neural_network = train_network(all_neural_network, all_optimizer, total_min_training_set, train_labels,
                                       criterion)

    print('Finished Training')

    total_wrong = 0
    for i in range(0, len(test_labels)):
        output = ten_neural_network(ten_min_test_set[i])

        total_wrong += abs(output.item() - test_labels[i])
        # print(total_wrong)
    print(total_wrong)

    total_wrong = 0
    for i in range(0, len(test_labels)):
        output = twenty_neural_network(twenty_min_test_set[i])
        total_wrong += abs(output.item() - test_labels[i])

    print(total_wrong)

    total_wrong = 0
    for i in range(0, len(test_labels)):
        output = all_neural_network(total_min_test_set[i])
        total_wrong += abs(output.item() - test_labels[i])

    print(total_wrong)


if __name__ == "__main__":
    main()
