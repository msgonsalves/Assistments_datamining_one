import csv
import pprint
import pickle
import os
import pandas
import ast
import re
import operator

import numpy as np
from collections import defaultdict
from scipy import stats


def grade(ans, given_ans, a_prob):

    if len(given_ans) > 0 or a_prob == 'VH134387':
        if 'selection' in given_ans:
            if given_ans['selection'] == ans[0]:
                return 1
            elif given_ans['selection'] == ans[1]:
                return 0
            else:
                return -1

        else:
            if len(given_ans) == 0:
                return 0
            tot_quest = len(given_ans)
            tot_correct = 0
            for a_key in given_ans:
                if given_ans[a_key][0] == ans[a_key][0]:
                    tot_correct += 1
                elif given_ans[a_key][0] == ans[a_key][1]:
                    tot_correct += 0
                else:
                    tot_correct-=0
            return tot_correct/tot_quest


    else:
        return 0

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def get_float(val_one, val_two):
    if val_two:
        return val_one / val_two
    else:
        return val_one


def problem_answers(id_map, answer):
    for a_prob in id_map:

        if 'selection' in answer[a_prob][2]:
            if answer[a_prob][2]['selection'] in id_map[a_prob]:
                id_map[a_prob][answer[a_prob][2]['selection']] += 1
            else:
                id_map[a_prob][answer[a_prob][2]['selection']] = 1
        else:
            for a_val in answer[a_prob][2]:
                if a_val in id_map[a_prob]:
                    if answer[a_prob][2][a_val][0] in id_map[a_prob][a_val]:
                        id_map[a_prob][a_val][answer[a_prob][2][a_val][0]] += 1
                    else:
                        id_map[a_prob][a_val][answer[a_prob][2][a_val][0]] = 1
                else:
                    id_map[a_prob][a_val] = {}
                    id_map[a_prob][a_val][answer[a_prob][2][a_val][0]] = 1


def parse_contentLaTex(clt):
    clt = re.sub('[$]', '', clt)
    total_string = clt.split('\\')
    real_string = ""
    for a_val in total_string:
        if 'mathrm' in a_val:
            next_val = re.sub(r'.*{', '', a_val)
            next_val = re.sub(r'}.*', '', next_val)
            real_string += next_val

        if a_val.isdigit():
            real_string += a_val
        else:
            if is_float(a_val):
                real_string += a_val

        if "frac" in a_val:
            first_val = a_val[a_val.find('{') + 1:a_val.find('}')]

            second_val = a_val[8:10]
            second_val = re.findall("\d+", second_val)
            if len(second_val) > 0:
                second_val = second_val[0]
            next_val = ""
            int_first_val = "test"
            try:
                int_first_val = float(first_val)
                int_second_val = float(second_val)
                next_val = str(int_first_val / int_second_val)
            except:
                if not int_first_val == "test":
                    next_val = str(get_float(int_first_val, 1))
                else:
                    next_val = "0"

            real_string += next_val

    return real_string


def parse_sfs(sfs, current_nums, nums):
    real_sfs = ast.literal_eval(sfs)
    if nums:
        place = real_sfs['numericIdentifier']
    else:
        place = real_sfs['partId']
    current_context = real_sfs['contentLaTeX']

    real_string = parse_contentLaTex(current_context)
    temp_string = real_string

    try:
        cursor_loc = current_nums[place][1]
    except:
        cursor_loc = 0

    if not place in current_nums:
        cursor_loc = len(temp_string)

    code = real_sfs['code']

    if len(temp_string) == cursor_loc:
        if "Digit" in code:
            digit = re.findall("\d+", code)
            current_nums[place] = [temp_string + digit[0], cursor_loc + 1]

        elif "Key" in code:
            current_nums[place] = [temp_string + code[-1], cursor_loc + 1]

        elif "Period" in code:
            current_nums[place] = [temp_string + ".", cursor_loc + 1]

        elif "Back" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string[:-1], cursor_loc - 1]

        elif "Tab" in code:
            pass
        elif "Left" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string, cursor_loc - 1]
        elif "Right" in code:
            pass
        elif "space" in code:
            current_nums[place] = [temp_string + " ", cursor_loc + 1]

        else:
            current_nums[place] = [temp_string + " ", cursor_loc + 1]
            pass
    else:
        if "Digit" in code:
            digit = re.findall("\d+", code)
            current_nums[place] = [temp_string[:cursor_loc] + digit[0] + temp_string[cursor_loc:], cursor_loc + 1]

        elif "Key" in code:
            current_nums[place] = [temp_string[:cursor_loc] + code[-1] + temp_string[cursor_loc:], cursor_loc + 1]

        elif "Period" in code:
            current_nums[place] = [temp_string[:cursor_loc] + "." + temp_string[cursor_loc:], cursor_loc + 1]

        elif "Back" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string[:cursor_loc - 1] + temp_string[cursor_loc:], cursor_loc - 1]

        elif "Tab" in code:
            pass
        elif "Left" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string, cursor_loc - 1]
        elif "Right" in code:
            current_nums[place] = [temp_string, cursor_loc + 1]
        elif "space" in code:
            current_nums[place] = [temp_string[:cursor_loc] + " " + temp_string[cursor_loc:], cursor_loc + 1]

        else:
            pass

    return current_nums


def make_full_maps(prob_ids, map, full):
    new_map = {}
    for a_prob in prob_ids:
        if a_prob in map:
            new_map[a_prob] = map[a_prob]
        else:
            new_map[a_prob] = [0, 0, {}]

    if full:
        problem_answers(prob_ids, new_map)
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

    # print(training_label)
    time_for_problems = []
    first_ten_minutes = []
    first_twenty_minutes = []
    alt_time_for_problems = []
    prob_ids = {}
    o_prob_ids = []

    total_time_for_problems = defaultdict(list)
    total_clicks_for_problems = defaultdict(list)
    ten_time_for_problems = defaultdict(list)
    ten_clicks_for_problems = defaultdict(list)
    twenty_time_for_problems = defaultdict(list)
    twenty_clicks_for_problems = defaultdict(list)

    old_student_id = student_ids[0]
    temp_student_id = old_student_id
    test_list = student_ids.copy()

    prob_id_answer_counts = {}

    for i in range(0, len(student_ids)):
        temp_prob_id = problem_ids[i]
        if not temp_prob_id in prob_ids:
            o_prob_ids.append(temp_prob_id)
            prob_ids[temp_prob_id] = {}
        if len(prob_ids) == 25:
            break

    start_time = 0

    temp_times = {}
    ten_temp = {}
    twenty_temp = {}

    temp_tot_time = 0
    num_clicks = 0
    current_fill_in = {}
    for i in range(0, len(student_ids)):
        temp_prob_id = problem_ids[i]

        temp_student_id = student_ids[i]

        num_clicks += 1

        if not temp_student_id == old_student_id:

            alt_time_for_problems.append([old_student_id, make_full_maps(prob_ids, temp_times, full=True)])
            first_ten_minutes.append([old_student_id, make_full_maps(prob_ids, ten_temp, full=False)])
            first_twenty_minutes.append([old_student_id, make_full_maps(prob_ids, twenty_temp, full=False)])

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
            old_student_id = temp_student_id

        if type_of_item[i] == 'MultipleFillInBlank' or type_of_item[i] == 'FillInBlank':
            if action[i] == 'Math Keypress':
                current_fill_in = parse_sfs(extended_info[i], current_fill_in, nums=True)

        elif type_of_item[i] == 'CompositeCR':
            if action[i] == 'Math Keypress':
                current_fill_in = parse_sfs(extended_info[i], current_fill_in, nums=False)

        if action[i] == 'Click Choice':
            selection = extended_info[i]
            choice = selection[selection.find('_') + 1: selection.find(':')]
            current_fill_in['selection'] = choice

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
                                              num_clicks + ten_temp[temp_prob_id][1],
                                              current_fill_in]
                else:
                    ten_temp[temp_prob_id] = [total_time, num_clicks, current_fill_in]

            if temp_tot_time < 1200:
                if temp_prob_id in twenty_temp:
                    twenty_temp[temp_prob_id] = [total_time + twenty_temp[temp_prob_id][0],
                                                 num_clicks + twenty_temp[temp_prob_id][1],
                                                 current_fill_in]
                else:
                    twenty_temp[temp_prob_id] = [total_time, num_clicks, current_fill_in]
            if total_time < 3599:
                if temp_prob_id in temp_times:
                    temp_times[temp_prob_id] = [total_time + temp_times[temp_prob_id][0],
                                                num_clicks + temp_times[temp_prob_id][1],
                                                current_fill_in]
                else:
                    temp_times[temp_prob_id] = [total_time, num_clicks, current_fill_in]

            # print(temp_student_id, temp_prob_id, current_fill_in)
            current_fill_in = {}
            start_time = end_time
            num_clicks = 0

    alt_time_for_problems.append([temp_student_id, make_full_maps(prob_ids, temp_times, full=True)])
    first_ten_minutes.append([temp_student_id, make_full_maps(prob_ids, ten_temp, full=False)])
    first_twenty_minutes.append([temp_student_id, make_full_maps(prob_ids, twenty_temp, full=False)])

    for a_num in ten_temp:
        ten_time_for_problems[a_num].append(ten_temp[a_num][0])
        ten_clicks_for_problems[a_num].append((ten_temp[a_num][1]))

    for a_num in twenty_temp:
        twenty_time_for_problems[a_num].append(twenty_temp[a_num][0])
        twenty_clicks_for_problems[a_num].append((twenty_temp[a_num][1]))

    for a_num in temp_times:
        total_time_for_problems[a_num].append(temp_times[a_num][0])
        total_clicks_for_problems[a_num].append((temp_times[a_num][1]))

    print("id: ", old_student_id, "  answers: ", temp_times)

    print(len(alt_time_for_problems), len(training_label))
    # print("testing: ", ten_time_for_problems)
    # print(first_ten_minutes[118])
    # print(first_twenty_minutes[118])
    # print(alt_time_for_problems[536])

    for a_prob in prob_ids:
        print("prov_id: ", a_prob, "answers: ", prob_ids[a_prob])

    file = open('num_answers.csv', 'w+')
    file.close()
    file = open('num_answers.csv', 'w', newline='')
    writer = csv.writer(file)
    prob_ans = {}
    for a_prob in prob_ids:
        first = True
        if len(prob_ids[a_prob]) > 1 or a_prob == 'VH134387' or a_prob == 'VH134373':
            for an_answer in prob_ids[a_prob]:
                if type(prob_ids[a_prob][an_answer]) == type(1):
                    sorted_x = sorted(prob_ids[a_prob].items(), key=operator.itemgetter(1), reverse=True)
                    row = [a_prob, "", "most common answer", sorted_x[0][0], sorted_x[0][1], "most common wrong answer",
                           sorted_x[1][0], sorted_x[1][1]]
                    prob_ans[a_prob] = [sorted_x[0][0], sorted_x[1][0]]
                    writer.writerow(row)
                    break
                elif type(prob_ids[a_prob][an_answer]) == type({}):
                    if first:
                        prob_ans[a_prob] = {}
                        first = False

                    for a_part in prob_ids[a_prob][an_answer]:
                        sorted_x = sorted(prob_ids[a_prob][an_answer].items(), key=operator.itemgetter(1), reverse=True)
                        row = [a_prob, "part: " + an_answer, "most common answer", sorted_x[0][0], sorted_x[0][1],
                               "most common wrong answer", sorted_x[1][0], sorted_x[1][1]]
                        writer.writerow(row)
                        prob_ans[a_prob][an_answer] = [sorted_x[0][0], sorted_x[1][0]]
                        break
        else:
            prob_ans[a_prob] = [0, 0]
    file.close()
    print(prob_ans)

    ten_temp_zscores_list = []
    twenty_temp_zscores_list = []
    tot_temp_zscores_list = []
    ten_temp_click_zscores_list = []
    twenty_temp_click_zscores_list = []
    tot_temp_click_zscores_list = []
    first_ten_minutes_zcore = []
    first_twenty_minutes_zcore = []
    ten_grade = []
    twenty_grade = []
    tot_grade = []
    tot_zscore = []
    for a_prob in prob_ids:
        ten_temp_regular_list = []
        twenty_temp_regular_list = []
        tot_temp_regular_list = []
        ten_temp_click_regular_list = []
        twenty_temp_click_regular_list = []
        tot_temp_click_regular_list = []
        ten_temp_grade = []
        twenty_temp_grade = []
        tot_temp_grade = []
        for i in range(0, len(first_ten_minutes)):
            ten_temp_regular_list.append(first_ten_minutes[i][1][a_prob][0])
            twenty_temp_regular_list.append(first_twenty_minutes[i][1][a_prob][0])
            tot_temp_regular_list.append(alt_time_for_problems[i][1][a_prob][0])
            ten_temp_click_regular_list.append(first_twenty_minutes[i][1][a_prob][1])
            twenty_temp_click_regular_list.append(first_ten_minutes[i][1][a_prob][1])
            tot_temp_click_regular_list.append(alt_time_for_problems[i][1][a_prob][1])
            ten_temp_grade.append(grade(prob_ans[a_prob], first_ten_minutes[i][1][a_prob][2], a_prob))
            twenty_temp_grade.append(grade(prob_ans[a_prob], first_twenty_minutes[i][1][a_prob][2], a_prob))
            tot_temp_grade.append(grade(prob_ans[a_prob], alt_time_for_problems[i][1][a_prob][2], a_prob))

        ten_grade.append(ten_temp_grade)
        twenty_grade.append(twenty_temp_grade)
        tot_grade.append(tot_temp_grade)
        ten_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(ten_temp_regular_list))))
        twenty_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(twenty_temp_regular_list))))
        tot_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(tot_temp_regular_list))))
        ten_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(ten_temp_click_regular_list))))
        twenty_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(twenty_temp_click_regular_list))))
        tot_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(tot_temp_click_regular_list))))

    pprint.pprint(len(ten_temp_zscores_list[0]))
    print(len(ten_temp_click_zscores_list[0]))
    print("zscore list: ", ten_temp_zscores_list)
    print("grades_list: ", tot_grade)
    print(prob_ans)

    print(type(ten_grade))
    for i in range(0, len(alt_time_for_problems)):
        ten_temp_zcores_map = {}
        twenty_temp_zcores_map = {}
        tot_temp_zcores_map = {}
        for j in range(0, len(prob_ids)):

            ten_temp_zcores_map[o_prob_ids[j]] = [ten_temp_zscores_list[j][i]/3, ten_temp_click_zscores_list[j][i]/3, ten_grade[j][i]]
            twenty_temp_zcores_map[o_prob_ids[j]] = [ten_temp_zscores_list[j][i]/3, ten_temp_click_zscores_list[j][i]/3, twenty_grade[j][i]]
            tot_temp_zcores_map[o_prob_ids[j]] = [ten_temp_zscores_list[j][i]/3, ten_temp_click_zscores_list[j][i]/3, tot_grade[j][i]]

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

    print(first_ten_minutes_zcore[0])
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
