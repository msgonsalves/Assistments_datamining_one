import csv
import pprint
import pickle
import os
import pandas
import ast
import re

import numpy as np
from collections import defaultdict
from scipy import stats


def parse_sfs(sfs, current_num):
    real_sfs = ast.literal_eval(sfs)

    code = real_sfs['code']
    if "Digit" in code:
        digit = re.findall("\d+", code)
        current_num += digit[0]
    elif "Period" in code:
        current_num += "."
    elif "Back" in code:
        current_num = current_num[:-1]
    else:
        current_num += " "

    return current_num

def make_full_maps(prob_ids, map):
    new_map = {}
    for a_prob in prob_ids:
        if a_prob in map:
            new_map[a_prob] = map[a_prob]
        else:
            new_map[a_prob] = [0, 0]

    return new_map


def append_to_dict(list, dict):
    for a_num in map:
        dict[a_num].append[map[a_num]]


def main():
    def convert_string_to_time(time_string):
        hours_minutes = time_string.split(":")
        return float(hours_minutes[0]) * 60 + float(hours_minutes[1])

    student_ids = []
    block = []
    problem_ids = []
    type_of_item = []
    action = []
    extended_info = []
    time_of_event = []

    label_student_ids = []
    training_label = []

    with open('data_a_train.csv')as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                student_ids.append(row[0])
                problem_ids.append(row[2])
                type_of_item.append(row[3])
                action.append(row[4])
                extended_info.append(row[5])
                time_of_event.append(row[6])

            line_count += 1

    with open('data_train_label.csv')as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                label_student_ids.append(row[1])
                if row[2] == 'True':
                    training_label.append([row[1], 1])
                else:
                    training_label.append([row[1], 0])
            line_count += 1

    print(training_label)
    time_for_problems = []
    first_ten_minutes = []
    first_twenty_minutes = []
    alt_time_for_problems = []
    prob_ids = []

    total_time_for_problems = defaultdict(list)
    total_clicks_for_problems = defaultdict(list)
    ten_time_for_problems = defaultdict(list)
    ten_clicks_for_problems = defaultdict(list)
    twenty_time_for_problems = defaultdict(list)
    twenty_clicks_for_problems = defaultdict(list)

    old_student_id = student_ids[0]
    temp_student_id = old_student_id
    test_list = student_ids.copy()

    for i in range(0, len(student_ids)):
        temp_prob_id = problem_ids[i]
        if not temp_prob_id in prob_ids:
            prob_ids.append(temp_prob_id)
        if len(prob_ids) == 25:
            break

    start_time = 0

    temp_times = {}
    ten_temp = {}
    twenty_temp = {}

    temp_tot_time = 0
    num_clicks = 0
    current_fill_in = ""
    for i in range(0, len(student_ids)):
        temp_prob_id = problem_ids[i]
        if not temp_prob_id in prob_ids:
            prob_ids.append(temp_prob_id)

        temp_student_id = student_ids[i]

        num_clicks += 1

        if not temp_student_id == old_student_id:

            alt_time_for_problems.append([old_student_id, make_full_maps(prob_ids, temp_times)])
            first_ten_minutes.append([old_student_id, make_full_maps(prob_ids, ten_temp)])
            first_twenty_minutes.append([old_student_id, make_full_maps(prob_ids, twenty_temp)])

            for a_num in ten_temp:
                ten_time_for_problems[a_num].append(ten_temp[a_num][0])
                ten_clicks_for_problems[a_num].append((ten_temp[a_num][1]))

            for a_num in twenty_temp:
                twenty_time_for_problems[a_num].append(twenty_temp[a_num][0])
                twenty_clicks_for_problems[a_num].append((twenty_temp[a_num][1]))

            for a_num in temp_times:
                total_time_for_problems[a_num].append(temp_times[a_num][0])
                total_clicks_for_problems[a_num].append((temp_times[a_num][1]))

            temp_times = {}
            ten_temp = {}
            twenty_temp = {}
            temp_tot_time = 0
            current_fill_in = ""
            old_student_id = temp_student_id

        if action[i] == 'Math Keypress':
            current_fill_in = parse_sfs(extended_info[i], current_fill_in)
            print(current_fill_in)

        if action[i] == 'Enter Item':
            start_time = convert_string_to_time(time_of_event[i])
            num_clicks = 0

        if action[i] == 'Exit Item':
            end_time = convert_string_to_time(time_of_event[i])
            if end_time > start_time:
                total_time = (convert_string_to_time(time_of_event[i]) - start_time) / 10
            else:
                total_time = (3600 - start_time) + end_time

            if total_time < 3599:
                time_for_problems.append([temp_student_id, temp_prob_id, total_time])
                temp_tot_time += total_time

            if temp_tot_time < 600:
                if temp_prob_id in ten_temp:
                    ten_temp[temp_prob_id] = [total_time + ten_temp[temp_prob_id][0],
                                              num_clicks + ten_temp[temp_prob_id][1]]
                else:
                    ten_temp[temp_prob_id] = [total_time, num_clicks]

            if temp_tot_time < 1200:
                if temp_prob_id in twenty_temp:
                    twenty_temp[temp_prob_id] = [total_time + twenty_temp[temp_prob_id][0],
                                                 num_clicks + twenty_temp[temp_prob_id][1]]
                else:
                    twenty_temp[temp_prob_id] = [total_time, num_clicks]
            if total_time < 3599:
                if temp_prob_id in temp_times:
                    temp_times[temp_prob_id] = [total_time + temp_times[temp_prob_id][0],
                                                num_clicks + temp_times[temp_prob_id][1]]
                else:
                    temp_times[temp_prob_id] = [total_time, num_clicks]

            start_time = end_time
            num_clicks = 0

    alt_time_for_problems.append([temp_student_id, make_full_maps(prob_ids, temp_times)])
    first_ten_minutes.append([temp_student_id, make_full_maps(prob_ids, ten_temp)])
    first_twenty_minutes.append([temp_student_id, make_full_maps(prob_ids, twenty_temp)])

    for a_num in ten_temp:
        ten_time_for_problems[a_num].append(ten_temp[a_num][0])
        ten_clicks_for_problems[a_num].append((ten_temp[a_num][1]))

    for a_num in twenty_temp:
        twenty_time_for_problems[a_num].append(twenty_temp[a_num][0])
        twenty_clicks_for_problems[a_num].append((twenty_temp[a_num][1]))

    for a_num in temp_times:
        total_time_for_problems[a_num].append(temp_times[a_num][0])
        total_clicks_for_problems[a_num].append((temp_times[a_num][1]))

    print(len(alt_time_for_problems), len(training_label))
    print(first_ten_minutes[118])
    print(first_twenty_minutes[118])
    print(alt_time_for_problems[536])

    ten_temp_zscores_list = []
    twenty_temp_zscores_list = []
    tot_temp_zscores_list = []
    ten_temp_click_zscores_list = []
    twenty_temp_click_zscores_list = []
    tot_temp_click_zscores_list = []
    first_ten_minutes_zcore = []
    first_twenty_minutes_zcore = []
    tot_zscore = []
    for a_prob in prob_ids:
        ten_temp_regular_list = []
        twenty_temp_regular_list = []
        tot_temp_regular_list = []
        ten_temp_click_regular_list = []
        twenty_temp_click_regular_list = []
        tot_temp_click_regular_list = []
        for i in range(0, len(first_ten_minutes)):
            ten_temp_regular_list.append(first_ten_minutes[i][1][a_prob][0])
            twenty_temp_regular_list.append(first_twenty_minutes[i][1][a_prob][0])
            tot_temp_regular_list.append(alt_time_for_problems[i][1][a_prob][0])
            ten_temp_click_regular_list.append(first_twenty_minutes[i][1][a_prob][1])
            twenty_temp_click_regular_list.append(first_ten_minutes[i][1][a_prob][1])
            tot_temp_click_regular_list.append(alt_time_for_problems[i][1][a_prob][1])

        ten_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(ten_temp_regular_list))))
        twenty_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(twenty_temp_regular_list))))
        tot_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(tot_temp_regular_list))))
        ten_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(ten_temp_click_regular_list))))
        twenty_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(twenty_temp_click_regular_list))))
        tot_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(tot_temp_click_regular_list))))

    pprint.pprint(len(ten_temp_zscores_list[0]))
    print(len(ten_temp_click_zscores_list[0]))

    for i in range(0, len(alt_time_for_problems)):
        ten_temp_zcores_map = {}
        twenty_temp_zcores_map = {}
        tot_temp_zcores_map = {}
        for j in range(0, len(prob_ids)):
            ten_temp_zcores_map[prob_ids[j]] = [ten_temp_zscores_list[j][i], ten_temp_click_zscores_list[j][i]]
            twenty_temp_zcores_map[prob_ids[j]] = [ten_temp_zscores_list[j][i], ten_temp_click_zscores_list[j][i]]
            tot_temp_zcores_map[prob_ids[j]] = [ten_temp_zscores_list[j][i], ten_temp_click_zscores_list[j][i]]

        first_ten_minutes_zcore.append([first_ten_minutes[i][0], ten_temp_zcores_map])
        first_twenty_minutes_zcore.append([first_twenty_minutes[i][0], ten_temp_zcores_map])
        tot_zscore.append([alt_time_for_problems[i][0], ten_temp_zcores_map])

    if os.path.exists('alt_time_for_problems_file'):
        os.remove('alt_time_for_problems_file')

    if os.path.exists('ten_minutes'):
        os.remove('ten_minutes')

    if os.path.exists('twenty_minutes'):
        os.remove('twenty_minutes')

    if os.path.exists('training_labels'):
        os.remove('training_labels')

    if os.path.exists('prob_ids'):
        os.remove('prob_ids')

    tot_zscore.sort()
    first_ten_minutes_zcore.sort()
    first_twenty_minutes_zcore.sort()
    training_label.sort()

    file = open("alt_time_for_problems_file", 'wb')
    pickle.dump(tot_zscore, file)
    file.close()

    file = open("ten_minutes", 'wb')
    pickle.dump(first_ten_minutes_zcore, file)
    file.close()

    file = open("twenty_minutes", 'wb')
    pickle.dump(first_twenty_minutes_zcore, file)
    file.close()

    file = open("training_labels", 'wb')
    pickle.dump(training_label, file)
    file.close()

    file = open("prob_ids", 'wb')
    pickle.dump(prob_ids, file)
    file.close()


if __name__ == "__main__":
    main()
