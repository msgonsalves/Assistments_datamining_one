import csv
import pprint
import pickle
import os
import pandas
import ast
import re
import operator
import math

import numpy as np
from collections import defaultdict
from scipy import stats

SELECTION = ['VH098810', 'VH098519', 'VH098808', 'VH098759', 'VH098740', 'VH098753', 'VH098783', 'VH098783', 'VH098812',
             'VH098839', 'VH098597', 'VH098556', 'VH098522', 'VH098779', 'VH098834', 'VH139047']
OTHER = ['HELPMAT8', 'SecTimeOut', 'VH356862', 'BlockRev', 'EOSTimeLft']

MULT_FILL_IN = ['VH134366', 'VH139196']

SINGLE_FILL_IN = ['VH134387', 'VH134373']


def grade(ans, given_ans, a_prob, extra_time, running_time):


    if a_prob in SELECTION:
        if 'selection' in given_ans:
            if given_ans['selection'] == ans[0]:
                return [3, running_time]
            elif given_ans['selection'] == ans[1]:
                return [1.5, running_time + (.5 * extra_time)]
            else:
                return [0, running_time + extra_time]
        else:
            return [-3, running_time]
    elif a_prob in OTHER:
        given_ans = {}
        return [0, running_time + extra_time]
    else:
        if len(given_ans) == 0:
            return [-3, running_time]
        tot_quest = 0
        tot_correct = 0
        z = 0
        for a_key in given_ans:
            if given_ans[a_key][0] == ans[a_key][0]:
                tot_correct += 3
            elif given_ans[a_key][0] == ans[a_key][1]:
                tot_correct += 1.5
            else:
                tot_correct -= 0
            tot_quest += 1
            z += 1
        if a_prob in MULT_FILL_IN:
            if a_prob == 'VH134366':
                for i in range(z, 5):
                    tot_correct -= 3
                    tot_quest += 1
            else:
                for i in range(z, 3):
                    tot_correct -= 3
                    tot_quest += 1
        return [tot_correct / tot_quest, (running_time + extra_time) * (tot_correct / tot_quest)]


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


def parse_sfs(sfs, current_nums, problem_id, nums, ):
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
            current_nums[place] = [temp_string + digit[0], cursor_loc + 1, problem_id]

        elif "Key" in code:
            current_nums[place] = [temp_string + code[-1], cursor_loc + 1, problem_id]

        elif "Period" in code:
            current_nums[place] = [temp_string + ".", cursor_loc + 1, problem_id]

        elif "Back" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string[:-1], cursor_loc - 1, problem_id]

        elif "Tab" in code:
            pass
        elif "Left" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string, cursor_loc - 1, problem_id]
        elif "Right" in code:
            pass
        elif "space" in code:
            current_nums[place] = [temp_string + " ", cursor_loc + 1, problem_id]

        else:
            current_nums[place] = [temp_string + " ", cursor_loc + 1, problem_id]
            pass
    else:
        if "Digit" in code:
            digit = re.findall("\d+", code)
            current_nums[place] = [temp_string[:cursor_loc] + digit[0] + temp_string[cursor_loc:], cursor_loc + 1,
                                   problem_id]

        elif "Key" in code:
            current_nums[place] = [temp_string[:cursor_loc] + code[-1] + temp_string[cursor_loc:], cursor_loc + 1,
                                   problem_id]

        elif "Period" in code:
            current_nums[place] = [temp_string[:cursor_loc] + "." + temp_string[cursor_loc:], cursor_loc + 1,
                                   problem_id]

        elif "Back" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string[:cursor_loc - 1] + temp_string[cursor_loc:], cursor_loc - 1,
                                       problem_id]

        elif "Tab" in code:
            pass
        elif "Left" in code:
            if cursor_loc > 0:
                current_nums[place] = [temp_string, cursor_loc - 1, problem_id]
        elif "Right" in code:
            current_nums[place] = [temp_string, cursor_loc + 1, problem_id]
        elif "space" in code:
            current_nums[place] = [temp_string[:cursor_loc] + " " + temp_string[cursor_loc:], cursor_loc + 1,
                                   problem_id]

        else:
            pass

    return current_nums


def make_full_maps(prob_ids, map, full):
    new_map = {}
    for a_prob in prob_ids:
        if a_prob in map:
            new_map[a_prob] = map[a_prob]
        else:
            new_map[a_prob] = [0, 0, {}, 0, 0, 0]
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

                if row[2] == 'TRUE':
                    training_label.append([row[1], 0])
                else:
                    training_label.append([row[1], 1])
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
    old_problem_id = 0
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

    start_calc_time = 0
    end_calc_time = 0
    draw_time = 0
    start_draw_time = 0
    end_draw_time = 0
    draw_on = False
    calc_time = 0
    temp_tot_time = 0
    num_clicks = 0
    calc_open = False
    current_fill_in = {}
    ten_nav_bar_clicked = []
    twenty_nav_bar_clicked = []
    tot_nav_bar_clicked = []
    indiv_ten_nav_bar_clicked = 0
    indiv_twenty_nav_bar_clicked = 0
    indiv_tot_nav_bar_clicked = 0
    time_lapsed = 0
    tot_start_time = convert_string_to_time(time_of_event[0])

    for i in range(0, len(student_ids)):
        temp_prob_id = problem_ids[i]

        temp_student_id = student_ids[i]

        num_clicks += 1

        if not temp_student_id == old_student_id:

            alt_time_for_problems.append([old_student_id, make_full_maps(prob_ids, temp_times, full=True)])
            first_ten_minutes.append([old_student_id, make_full_maps(prob_ids, ten_temp, full=False)])
            first_twenty_minutes.append([old_student_id, make_full_maps(prob_ids, twenty_temp, full=False)])
            if old_student_id == '2333000033':
                print("anotha on: ", alt_time_for_problems[-1])
            for a_num in ten_temp:
                ten_time_for_problems[a_num].append(ten_temp[a_num][0])
                ten_clicks_for_problems[a_num].append((ten_temp[a_num][1]))

            for a_num in twenty_temp:
                twenty_time_for_problems[a_num].append(twenty_temp[a_num][0])
                twenty_clicks_for_problems[a_num].append((twenty_temp[a_num][1]))

            for a_num in temp_times:
                total_time_for_problems[a_num].append(temp_times[a_num][0])
                total_clicks_for_problems[a_num].append((temp_times[a_num][1]))

            ten_nav_bar_clicked.append(indiv_ten_nav_bar_clicked)
            twenty_nav_bar_clicked.append(indiv_twenty_nav_bar_clicked)
            tot_nav_bar_clicked.append(indiv_tot_nav_bar_clicked)
            temp_times = {}
            ten_temp = {}
            twenty_temp = {}
            old_student_id = temp_student_id
            time_lapsed = 0
            indiv_ten_nav_bar_clicked = 0
            indiv_twenty_nav_bar_clicked = 0
            indiv_tot_nav_bar_clicked = 0
            tot_start_time = convert_string_to_time(time_of_event[i])

        if type_of_item[i] == 'MultipleFillInBlank' or type_of_item[i] == 'FillInBlank':
            if action[i] == 'Math Keypress':
                current_fill_in = parse_sfs(extended_info[i], current_fill_in, temp_prob_id, nums=True)

        elif type_of_item[i] == 'CompositeCR':
            if action[i] == 'Math Keypress':
                current_fill_in = parse_sfs(extended_info[i], current_fill_in, temp_prob_id, nums=False)

        if action[i] == "Click Progress Navigator":
            current_time = convert_string_to_time(time_of_event[i])

            if current_time > tot_start_time:
                time_lapsed = (end_draw_time - tot_start_time)
            elif current_time == tot_start_time:
                print("something went very wrong", temp_student_id)
            else:
                time_lapsed = (3600 - tot_start_time) + current_time

            if time_lapsed < 600:
                indiv_ten_nav_bar_clicked += 1
            if time_lapsed < 1200:
                indiv_twenty_nav_bar_clicked += 1
            if time_lapsed < 2000:
                indiv_tot_nav_bar_clicked += 1

        if action[i] == "Open Calculator":
            if draw_on:
                end_draw_time = convert_string_to_time(time_of_event[i])

                if end_draw_time > start_draw_time:
                    draw_time += (end_draw_time - start_draw_time)
                elif end_draw_time == start_draw_time:
                    pass
                else:
                    draw_time += (3600 - start_draw_time) + end_draw_time
                draw_on = False

            calc_open = True
            start_calc_time = convert_string_to_time(time_of_event[i])

        elif action[i] == "Close Calculator":
            end_calc_time = convert_string_to_time(time_of_event[i])
            calc_open = False

            if end_calc_time > start_calc_time:
                calc_time += (end_calc_time - start_calc_time)
            elif end_calc_time == start_calc_time:
                pass
            else:
                calc_time += (3600 - start_calc_time) + end_calc_time

        if action[i] == 'Scratchwork Mode On':
            if calc_open:
                end_calc_time = convert_string_to_time(time_of_event[i])

                if end_calc_time > start_calc_time:
                    calc_time += (end_calc_time - start_calc_time)
                elif end_calc_time == start_calc_time:
                    pass
                else:
                    calc_time += (3600 - start_calc_time) + end_calc_time

                calc_open = False

            start_draw_time = convert_string_to_time(time_of_event[i])
            draw_on = True

        if action[i] == 'Scratchwork Mode Off':
            end_draw_time = convert_string_to_time(time_of_event[i])
            draw_on = False

            if end_draw_time > start_draw_time:
                draw_time += (end_draw_time - start_draw_time)
            elif end_draw_time == start_draw_time:
                pass
            else:
                draw_time += (3600 - start_draw_time) + end_draw_time

        if action[i] == 'Click Choice':
            selection = extended_info[i]
            choice = selection[selection.find('_') + 1: selection.find(':')]
            current_fill_in['selection'] = choice

        if action[i] == 'Enter Item':
            start_time = convert_string_to_time(time_of_event[i])
            num_clicks = 0
            old_problem_id = temp_prob_id
            try:
                current_fill_in = temp_times[temp_prob_id][2]
            except:
                current_fill_in = {}

        if action[i] == 'Exit Item' and old_problem_id == temp_prob_id:
            end_time = convert_string_to_time(time_of_event[i])
            temp_tot_time = 0
            if end_time > tot_start_time:
                temp_tot_time = end_time - tot_start_time

            elif end_time == tot_start_time:
                "why"
            else:
                temp_tot_time = (3600 - tot_start_time) + end_time

            if end_time > start_time:
                total_time = (convert_string_to_time(time_of_event[i]) - start_time)
            elif end_time == start_time:
                total_time = total_time
            else:
                total_time = (3600 - start_time) + end_time

            if temp_student_id == '2333000033':
                print(temp_tot_time)

            if calc_open:
                if end_time > start_calc_time:
                    calc_time += (end_time - start_calc_time)
                elif end_time == start_calc_time:
                    pass
                else:
                    calc_time += (3600 - start_calc_time) + end_time

            if draw_on:
                if end_time > start_draw_time:
                    draw_time += (end_time - start_draw_time)
                elif end_time == start_draw_time:
                    pass
                else:
                    draw_time += (3600 - start_draw_time) + end_time

            if total_time < 3599:
                time_for_problems.append([temp_student_id, temp_prob_id, total_time])

            if temp_tot_time < 600:
                if temp_prob_id in ten_temp:
                    ten_temp[temp_prob_id] = [total_time + ten_temp[temp_prob_id][0],
                                              num_clicks + ten_temp[temp_prob_id][1],
                                              current_fill_in,
                                              calc_time + ten_temp[temp_prob_id][3],
                                              draw_time + ten_temp[temp_prob_id][4],
                                              ten_temp[temp_prob_id][5] + 1]
                else:
                    ten_temp[temp_prob_id] = [total_time, num_clicks, current_fill_in, calc_time, draw_time, 0]

            if temp_tot_time < 1200:
                if temp_prob_id in twenty_temp:
                    twenty_temp[temp_prob_id] = [total_time + twenty_temp[temp_prob_id][0],
                                                 num_clicks + twenty_temp[temp_prob_id][1],
                                                 current_fill_in,
                                                 calc_time + twenty_temp[temp_prob_id][3],
                                                 draw_time + twenty_temp[temp_prob_id][4],
                                                 twenty_temp[temp_prob_id][5] + 1]
                else:
                    twenty_temp[temp_prob_id] = [total_time, num_clicks, current_fill_in, calc_time, draw_time, 0]
            if total_time < 3599:
                if temp_prob_id in temp_times:
                    temp_times[temp_prob_id] = [total_time + temp_times[temp_prob_id][0],
                                                num_clicks + temp_times[temp_prob_id][1],
                                                current_fill_in,
                                                calc_time + temp_times[temp_prob_id][3],
                                                draw_time + temp_times[temp_prob_id][4],
                                                temp_times[temp_prob_id][5] + 1]
                else:
                    temp_times[temp_prob_id] = [total_time, num_clicks, current_fill_in, calc_time, draw_time, 0]

            current_fill_in = {}
            calc_open = False
            draw_on = False
            calc_time = 0
            draw_time = 0
            start_time = end_time
            num_clicks = 0

    ten_nav_bar_clicked.append(indiv_ten_nav_bar_clicked)
    twenty_nav_bar_clicked.append(indiv_twenty_nav_bar_clicked)
    tot_nav_bar_clicked.append(indiv_tot_nav_bar_clicked)
    print(len(ten_nav_bar_clicked))
    print(ten_nav_bar_clicked)
    alt_time_for_problems.append([temp_student_id, make_full_maps(prob_ids, temp_times, full=True)])
    first_ten_minutes.append([temp_student_id, make_full_maps(prob_ids, ten_temp, full=False)])
    first_twenty_minutes.append([temp_student_id, make_full_maps(prob_ids, twenty_temp, full=False)])

    alt_time_for_problems.sort()
    first_ten_minutes.sort()
    first_twenty_minutes.sort()
    training_label.sort()

    # for i in range(0, len(alt_time_for_problems)):
    #     print(training_label[i][0], alt_time_for_problems[i][0])
    #     if not training_label[i][0] == alt_time_for_problems[i][0]:
    #         print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", i)

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

    print(first_ten_minutes[0])
    print(first_twenty_minutes[0])
    print(alt_time_for_problems[0])
    # pprint.pprint(first_ten_minutes)
    print(len(alt_time_for_problems), len(training_label))
    # print("testing: ", ten_time_for_problems)
    # print(first_ten_minutes[118])
    # print(first_twenty_minutes[118])
    # print(alt_time_for_problems[536])

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
    j_ten_time = []
    j_twenty_time = []
    j_tot_time = []
    ten_grade = []
    twenty_grade = []
    tot_grade = []
    tot_zscore = []
    ten_ans = []
    twenty_ans = []
    tot_ans = []
    ten_calc_time = []
    twenty_calc_time = []
    tot_calc_time = []
    j_ten_c_time = []
    j_twenty_c_time = []
    j_tot_c_time = []
    ten_draw_time = []
    twenty_draw_time = []
    tot_draw_time = []
    j_ten_d_time = []
    j_twenty_d_time = []
    j_tot_d_time = []
    ten_wrong_time = []
    twenty_wrong_time = []
    tot_wrong_time = []

    ten_rushed = []
    twenty_rushed = []
    tot_rushed = []

    ten_revisits = []
    twenty_revisits = []
    tot_revisits = []

    z_all_ten_calc = []
    z_all_twenty_calc = []
    z_all_tot_calc = []

    for i in range(0, len(first_ten_minutes)):
        temp_ten_wrong_time = 0
        temp_twenty_wrong_time = 0
        temp_tot_wrong_time = 0

        a_ten_calc = 0
        a_twenty_calc = 0
        a_tot_calc = 0

        for a_prob in prob_ids:
            temp_grade, temp_ten_wrong_time = grade(prob_ans[a_prob], first_ten_minutes[i][1][a_prob][2], a_prob,
                                                    first_ten_minutes[i][1][a_prob][0], temp_ten_wrong_time)

            temp_grade, temp_twenty_wrong_time = grade(prob_ans[a_prob], first_twenty_minutes[i][1][a_prob][2], a_prob,
                                                       first_twenty_minutes[i][1][a_prob][0], temp_twenty_wrong_time)

            temp_grade, temp_tot_wrong_time = grade(prob_ans[a_prob], alt_time_for_problems[i][1][a_prob][2], a_prob,
                                                    alt_time_for_problems[i][1][a_prob][0], temp_tot_wrong_time)

            a_ten_calc += first_ten_minutes[i][1][a_prob][3]
            a_twenty_calc += first_twenty_minutes[i][1][a_prob][3]
            a_tot_calc += first_twenty_minutes[i][1][a_prob][3]

        ten_wrong_time.append(temp_ten_wrong_time)
        twenty_wrong_time.append(temp_twenty_wrong_time)
        tot_wrong_time.append(temp_tot_wrong_time)

        z_all_ten_calc.append(a_ten_calc)
        z_all_twenty_calc.append(a_twenty_calc)
        z_all_tot_calc.append(a_tot_calc)

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
        ten_temp_ans = []
        twenty_temp_ans = []
        tot_temp_ans = []
        temp_ten_calc_time = []
        temp_twenty_calc_time = []
        temp_tot_calc_time = []
        temp_ten_draw_time = []
        temp_twenty_draw_time = []
        temp_tot_draw_time = []
        temp_ten_revisits = []
        temp_twenty_revisits = []
        temp_tot_revisits = []

        for i in range(0, len(first_ten_minutes)):
            ten_temp_regular_list.append(first_ten_minutes[i][1][a_prob][0])
            twenty_temp_regular_list.append(first_twenty_minutes[i][1][a_prob][0])
            tot_temp_regular_list.append(alt_time_for_problems[i][1][a_prob][0])
            ten_temp_click_regular_list.append(first_ten_minutes[i][1][a_prob][1])
            twenty_temp_click_regular_list.append(first_twenty_minutes[i][1][a_prob][1])
            tot_temp_click_regular_list.append(alt_time_for_problems[i][1][a_prob][1])

            temp_ten_revisits.append(first_ten_minutes[i][1][a_prob][5])
            temp_twenty_revisits.append(first_twenty_minutes[i][1][a_prob][5])
            temp_tot_revisits.append(alt_time_for_problems[i][1][a_prob][5])

            temp_grade, temp_ten_wrong_time = grade(prob_ans[a_prob], first_ten_minutes[i][1][a_prob][2], a_prob,
                                                    first_ten_minutes[i][1][a_prob][0], 0)
            ten_temp_grade.append(temp_grade)

            temp_grade, temp_twenty_wrong_time = grade(prob_ans[a_prob], first_twenty_minutes[i][1][a_prob][2], a_prob,
                                                       first_twenty_minutes[i][1][a_prob][0], 0)
            twenty_temp_grade.append(temp_grade)
            temp_grade, temp_tot_wrong_time = grade(prob_ans[a_prob], alt_time_for_problems[i][1][a_prob][2], a_prob,
                                                    alt_time_for_problems[i][1][a_prob][0], 0)

            tot_temp_grade.append(temp_grade)

            if alt_time_for_problems[i][0] == '2333000033':
                print("___________________: ", alt_time_for_problems[i][1][a_prob][2], a_prob)

            ten_temp_ans.append(first_ten_minutes[i][1][a_prob][2])
            twenty_temp_ans.append(first_twenty_minutes[i][1][a_prob][2])
            tot_temp_ans.append(alt_time_for_problems[i][1][a_prob][2])

            temp_ten_calc_time.append(first_ten_minutes[i][1][a_prob][3])
            temp_twenty_calc_time.append(first_twenty_minutes[i][1][a_prob][3])
            temp_tot_calc_time.append(alt_time_for_problems[i][1][a_prob][3])

            temp_ten_draw_time.append(first_ten_minutes[i][1][a_prob][4])
            temp_twenty_draw_time.append(first_twenty_minutes[i][1][a_prob][4])
            temp_tot_draw_time.append(alt_time_for_problems[i][1][a_prob][4])

        ten_ans.append(ten_temp_ans)
        twenty_ans.append(twenty_temp_ans)
        tot_ans.append(tot_temp_ans)

        j_ten_c_time.append(temp_ten_calc_time)
        j_twenty_c_time.append(temp_twenty_calc_time)
        j_tot_c_time.append(temp_tot_calc_time)



        j_ten_d_time.append(temp_ten_draw_time)
        j_twenty_d_time.append(temp_twenty_draw_time)
        j_tot_d_time.append(temp_tot_draw_time)

        ten_grade.append(ten_temp_grade)
        twenty_grade.append(twenty_temp_grade)
        tot_grade.append(tot_temp_grade)

        ten_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(ten_temp_regular_list))))
        twenty_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(twenty_temp_regular_list))))
        tot_temp_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(tot_temp_regular_list))))

        ten_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(ten_temp_click_regular_list))))
        twenty_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(twenty_temp_click_regular_list))))
        tot_temp_click_zscores_list.append(np.ndarray.tolist(stats.zscore(np.array(tot_temp_click_regular_list))))

        ten_calc_time.append(np.ndarray.tolist(stats.zscore(np.array(temp_ten_calc_time))))
        twenty_calc_time.append(np.ndarray.tolist(stats.zscore(np.array(temp_twenty_calc_time))))
        tot_calc_time.append(np.ndarray.tolist(stats.zscore(np.array(temp_tot_calc_time))))

        ten_draw_time.append(np.ndarray.tolist(stats.zscore(np.array(temp_ten_draw_time))))
        twenty_draw_time.append(np.ndarray.tolist(stats.zscore(np.array(temp_twenty_draw_time))))
        tot_draw_time.append(np.ndarray.tolist(stats.zscore(np.array(temp_tot_draw_time))))

        ten_rushed.append(np.percentile(np.array(ten_temp_regular_list), 5))
        twenty_rushed.append(np.percentile(np.array(twenty_temp_regular_list), 5))
        tot_rushed.append(np.percentile(np.array(tot_temp_regular_list), 5))

        ten_revisits.append(temp_ten_revisits)
        twenty_revisits.append(temp_twenty_revisits)
        tot_revisits.append(temp_tot_revisits)

        j_ten_time.append(ten_temp_regular_list)
        j_twenty_time.append(twenty_temp_regular_list)
        j_tot_time.append(tot_temp_regular_list)

    z_ten_calc_time = np.ndarray.tolist(stats.zscore(np.array(z_all_ten_calc)))
    z_twenty_calc_time = np.ndarray.tolist(stats.zscore(np.array(z_all_twenty_calc)))
    z_tot_calc_time = np.ndarray.tolist(stats.zscore(np.array(z_all_tot_calc)))

    print("total calc time; ", z_tot_calc_time, len(z_tot_calc_time))

    ten_wrong_time_zscore = np.ndarray.tolist(stats.zscore(np.array(ten_wrong_time)))
    twenty_wrong_time_zxore = np.ndarray.tolist(stats.zscore(np.array(twenty_wrong_time)))
    tot_wrong_time_zscore = np.ndarray.tolist(stats.zscore(np.array(tot_wrong_time)))

    other_file = open('actual_ans.csv', 'w+', newline='')
    other_writer = csv.writer(other_file)

    print(len(ten_ans[0]), len(first_ten_minutes))
    # pprint.pprint(ten_ans)
    first = True
    init_row = []
    for i in range(0, len(first_ten_minutes) + 1):
        if first:
            init_row = []
            init_row.append('student id')
            first = False
            for a_prob in prob_ids:
                if a_prob in SELECTION:
                    init_row.append(a_prob)
                elif a_prob in OTHER:
                    init_row.append(a_prob)
                else:
                    for an_answer in prob_ids[a_prob]:
                        for a_part in prob_ids[a_prob][an_answer]:
                            init_row.append(a_prob + " " + an_answer)
                            break
            other_writer.writerow(init_row)
        else:
            row = []
            row.append(first_ten_minutes[i - 1][0])
            first_multi = True
            second_multi = True
            k = 0
            for j in range(0, len(init_row) - 1):
                if first_multi or not ('VH134366' in init_row[j + 1]):
                    if second_multi or not ('VH139196' in init_row[j + 1]):
                        if init_row[j + 1] in SELECTION:
                            if len(tot_ans[k][i - 1]) > 0:

                                row.append(tot_ans[k][i - 1]['selection'])
                                k += 1
                            else:
                                row.append(-1)
                                k += 1
                        elif init_row[j + 1] in OTHER:
                            row.append(-1)
                            k += 1
                        elif 'VH134366' in init_row[j + 1] or 'VH139196' in init_row[j + 1]:
                            if first_multi:
                                first_multi = False
                            else:
                                second_multi = False
                            z = 0
                            for an_answer in tot_ans[k][i - 1]:
                                row.append(tot_ans[k][i - 1][an_answer][0])
                                z += 1
                            if second_multi:
                                for y in range(z, 5):
                                    row.append(-1)
                            else:
                                for y in range(z, 3):
                                    row.append(-1)
                            k += 1
                        elif 'VH134373' in init_row[j + 1] or 'VH134387' in init_row[j + 1]:
                            if len(tot_ans[k][i - 1]) > 0:
                                for an_answer in tot_ans[k][i - 1]:
                                    row.append(tot_ans[k][i - 1][an_answer][0])
                            else:
                                row.append(-1)
                            k += 1
                        else:
                            pass

            other_writer.writerow(row)
        # print(row)
    other_file.close()
    first = True
    file = open('calc_and_draw_times.csv', 'w+', newline='')
    writer = csv.writer(file)

    for i in range(0, len(first_ten_minutes) + 1):
        if first:
            row = []
            row.append('student id')
            first = False
            for a_prob in prob_ids:
                row.append(a_prob + " calc_time")
                row.append(a_prob + " sketch_time")
        else:
            row = []
            row.append(first_ten_minutes[i - 1][0])

            for j in range(0, len(prob_ids)):
                row.append(j_tot_c_time[j][i - 1])
                row.append(j_tot_d_time[j][i - 1])

        writer.writerow(row)
        # print(row)

    file.close()
    pprint.pprint(len(ten_temp_zscores_list[0]))
    print(ten_temp_click_zscores_list[0])
    print(twenty_temp_click_zscores_list[0])
    # print("zscore list: ", ten_temp_zscores_list)
    # print("grades_list: ", tot_grade)
    # print(prob_ans)

    for i in range(0, len(ten_calc_time)):
        for j in range(len(ten_calc_time[i])):
            if math.isnan(ten_calc_time[i][j]):
                ten_calc_time[i][j] = 0
            if math.isnan(twenty_calc_time[i][j]):
                twenty_calc_time[i][j] = 0
            if math.isnan(tot_calc_time[i][j]):
                tot_calc_time[i][j] = 0
            if math.isnan(ten_draw_time[i][j]):
                ten_draw_time[i][j] = 0
            if math.isnan(twenty_draw_time[i][j]):
                twenty_draw_time[i][j] = 0
            if math.isnan(tot_draw_time[i][j]):
                tot_draw_time[i][j] = 0
            if math.isnan(ten_temp_zscores_list[i][j]):
                ten_temp_zscores_list[i][j] = 0
            if math.isnan(ten_temp_click_zscores_list[i][j]):
                ten_temp_click_zscores_list[i][j] = 0
            if math.isnan(twenty_temp_zscores_list[i][j]):
                twenty_temp_zscores_list[i][j] = 0
            if math.isnan(twenty_temp_click_zscores_list[i][j]):
                twenty_temp_click_zscores_list[i][j] = 0
            if math.isnan(tot_temp_zscores_list[i][j]):
                tot_temp_zscores_list[i][j] = 0
            if math.isnan(tot_temp_click_zscores_list[i][j]):
                tot_temp_click_zscores_list[i][j] = 0

    for a_list in twenty_draw_time:
        for a_val in a_list:
            if math.isnan(a_val):
                print("how")
    print(type(ten_grade))
    print(len(ten_wrong_time_zscore), len(alt_time_for_problems))

    print(ten_temp_click_zscores_list[0])
    print(twenty_temp_click_zscores_list[0])
    tot_flagged = []
    for i in range(0, len(alt_time_for_problems)):
        ten_temp_zcores_map = {}
        twenty_temp_zcores_map = {}
        tot_temp_zcores_map = {}
        tot_flagged.append([1, alt_time_for_problems[i][0]])
        for j in range(0, len(prob_ids)):
            if not o_prob_ids[j] in OTHER:
                if ten_rushed[j] > j_ten_time[j][i]:
                    ten_flag = 0
                else:
                    ten_flag = 1
                if twenty_rushed[j] > j_twenty_time[j][i]:
                    twenty_flag = 0
                else:
                    twenty_flag = 1
                if tot_rushed[j] > j_tot_time[j][i]:
                    tot_flag = 0
                    tot_flagged[i] = [0, alt_time_for_problems[i][0]]
                else:
                    tot_flag = 1
            else:
                ten_flag = 0
                twenty_flag = 0
                tot_flag = 0
            ten_temp_zcores_map[o_prob_ids[j]] = [ten_temp_zscores_list[j][i], ten_temp_click_zscores_list[j][i],
                                                  ten_grade[j][i], ten_calc_time[j][i], ten_draw_time[j][i],
                                                  ten_revisits[j][i], ten_flag]
            twenty_temp_zcores_map[o_prob_ids[j]] = [twenty_temp_zscores_list[j][i],
                                                     twenty_temp_click_zscores_list[j][i],
                                                     twenty_grade[j][i], twenty_calc_time[j][i], twenty_draw_time[j][i],
                                                     twenty_revisits[j][i], twenty_flag]
            tot_temp_zcores_map[o_prob_ids[j]] = [tot_temp_zscores_list[j][i], tot_temp_click_zscores_list[j][i],
                                                  tot_grade[j][i], tot_calc_time[j][i], tot_draw_time[j][i],
                                                  tot_revisits[j][i], tot_flag]

        first_ten_minutes_zcore.append(
            [first_ten_minutes[i][0], ten_temp_zcores_map, ten_wrong_time_zscore[i], ten_nav_bar_clicked[i],
             z_ten_calc_time[i]])
        first_twenty_minutes_zcore.append(
            [first_twenty_minutes[i][0], twenty_temp_zcores_map, twenty_wrong_time_zxore[i], twenty_nav_bar_clicked[i],
             z_twenty_calc_time[i]])
        tot_zscore.append(
            [alt_time_for_problems[i][0], tot_temp_zcores_map, tot_wrong_time_zscore[i], tot_nav_bar_clicked[i],
             z_tot_calc_time[i]])

    print(first_ten_minutes_zcore[0])
    print(first_twenty_minutes_zcore[0])
    print(type(first_ten_minutes_zcore[0][1]['VH098556'][4]))
    print(first_ten_minutes_zcore[0][1]['VH098556'][4])
    efficient_count = 0
    efficient_people = []
    flagged_people = []
    for i in range(0, len(training_label)):
        if training_label[i][1] == 1:
            efficient_count += 1
            efficient_people.append(1)
        else:
            efficient_people.append(0)
        if tot_flagged[i][0] == 1:
            flagged_people.append(1)
        else:
            flagged_people.append(0)
    print(efficient_count)

    t_test = stats.ttest_ind(flagged_people, efficient_people)

    file = open('rushed_part_a.csv', 'w+', newline='')
    writer = csv.writer(file)
    writer.writerow(["rush", "STUDENTID"])
    for a_person in tot_flagged:
        writer.writerow(a_person)

    file.close()
    num_diff = 0
    num_diff_one = 0
    num_diff_zero = 0
    efficient_count = 0
    for i in range(0, len(tot_flagged)):
        # print(tot_flagged[i][1], training_label[i][0])
        if not flagged_people[i] == efficient_people[i]:
            num_diff += 1
            if efficient_people[i] == 0:
                num_diff_zero += 1
            else:
                num_diff_one += 1
            if flagged_people[i] == 1:
                efficient_count += 1
    print(num_diff, num_diff_one, num_diff_zero)
    print(efficient_count)
    print(t_test)
    print(len(tot_flagged), len(efficient_people), len(training_label))
    print(stats.pearsonr(flagged_people, efficient_people))
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

    print(first_ten_minutes_zcore[1])
    print(first_twenty_minutes_zcore[1])
    print(tot_zscore[1])
    print(training_label[1])
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
