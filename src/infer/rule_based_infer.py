from vietnam_number import n2w, w2n, w2n_single
import numpy as np

from ..utils.preprocess import preprocess


def check_float(value):
    if '\n' in value or '.' in value:
        return False
    try:
        if value[-1] == ',':
            return False
    except:
        pass
    try:
        new_value = value.replace(",", ".")
        new_value = float(new_value)
        return True
    except:
        return False


def conver_one_digit_to_text(value, index):
    output = digits[value]
    if value == "0":
        output = output[index]
        if index == 2:
            output =  output + " " + don_vi[str(index)]
    else:
        output =  output + " " + don_vi[str(index)]
    return output


def convert_int_with_below_three_digits_to_text(value):
    len_value = len(value)
    output = [conver_one_digit_to_text(value, len_value - index - 1) for index, value in enumerate(value)]
    new_output = " ".join(output)
    check_output = True
    zero_digits_description = digits["0"]
    remove_len = 0
    for index, val in enumerate(output[::-1]):
        #print(index)
        #print(zero_digits_description[index] + (" trăm" if index == 2 else ""))
        check_output = check_output and (val == zero_digits_description[index] + (" trăm" if index == 2 else ""))
        if check_output:
            remove_len += 1
            #if len_value == index + 1:
                #new_output = "không"
                #break
            #else:
                #remove_len += 1
    #print(remove_len)
    #if remove_len > 1 and new_output != "không":
    if remove_len > 0:
        new_output = " ".join(output[:-remove_len])
    if len_value > 1 and value[-1] == "4":
        new_output = new_output.replace("bốn đơn vị", "tư đơn vị")
    if len_value > 1 and value[-1] == "5":
        new_output = new_output.replace("năm đơn vị", "lăm đơn vị").replace("linh lăm", "linh năm")
    return new_output.replace("một mươi", "mười")


def convert_int_value_to_text(value, before = True):
    zero_value = ""
    index = 0
    for elem in value:
        if elem != "0":
            break
        else:
            index += 1
            if not before:
                zero_value += "không "
    if index == len(value):
        return ""
            
    new_value = value[index:]
    len_value = len(new_value)
    len_value_num = len_value // 3 + (0 if len_value % 3 == 0 else 1)
    reverse_value = value[::-1]
    true_value = [convert_int_with_below_three_digits_to_text(reverse_value[index * 3 : (index + 1) * 3][::-1])
                  + ("" if index == 0 else " " + don_vi_to[str(index)])
                  for index in range(len_value_num)]
    true_value = " ".join(true_value[::-1])
    return zero_value + true_value


def convert_int_value_to_text(value):
    str_value = str(value)
    str_result = n2w(str_value.replace(",", "."))
    if str_result[-11:] == ' không trăm':
        str_result = str_result[:-11]
    return str_result.replace("lẽ", "linh")


def convert_float_value_to_text(value):
    if not check_float(value):
        return None
    new_value = value.split(',')
    true_value = convert_int_value_to_text(new_value[0])
    if len(new_value) > 1:
        value_after = convert_int_value_to_text(new_value[1])
        true_value += (" phẩy " + value_after if value_after != "" else value_after)
    return true_value.replace(" đơn vị", "")
don_vi = {"2":"trăm", "1":"mươi", "0":"đơn vị"}
don_vi_to = {"1":"nghìn", "2":"triệu", "3":"tỷ"}
digits = {"0":["", "linh", "không"], "1":"một", "2":"hai", "3":"ba", "4":"bốn","5":"năm", "6":"sáu","7":"bảy",
          "8":"tám", "9":"chín"}


def check_having_don_vi(value):
    for key in don_vi_do_dai.keys():
        if key in value:
            full_num = value.split(key)
            if not check_float(full_num[0]):
                continue
            return True
    return False


def convert_float_having_don_vi_to_text(value):
    for key in don_vi_do_dai.keys():
        if key in value:
            full_num = value.split(key)
            if not check_float(full_num[0]):
                continue
            full_num[0] = convert_float_value_to_text(full_num[0])
            try:
                tail_used = " " + tails[full_num[1]]
            except:
                tail_used = ""
            return full_num[0] +  " " + don_vi_do_dai[key] + tail_used
    return False


tails = {'2':"vuông", '3':"khối"}
don_vi_do_dai = {"cm":"xăng ti mét", "dm":"đề xi mét", "m":"mét", "km":"ki lô mét", "mm":"mi li mét"}


def convert_txt_phan_so_to_float(split_choice_2):
    
    split_choice_2 = split_choice_2.split('}{')
    mau_so_choice = split_choice_2[1].split('}')[0]
    tu_so_choice = split_choice_2[0].split('{')[1]
    choice_result = float(tu_so_choice) / float(mau_so_choice)
    return choice_result


def convert_text_to_float(split_question_6):
    if 'phẩy' not in split_question_6:
        question_result = w2n(split_question_6)
    if 'phẩy' in split_question_6:
        split_question_6 = split_question_6.split('phẩy')
        question_result_before = str(w2n(split_question_6[0]))
        question_result_after = str(w2n_single(split_question_6[1]))
        question_result = question_result_before + '.' + question_result_after
        question_result = float(question_result)
    if 'phần' in split_question_6:
        split_question_6 = split_question_6.split('phần')
        question_result_before = str(w2n(split_question_6[0]))
        question_result_after = str(w2n(split_question_6[1]))
        return float(question_result_before)/float(question_result_after)
    return question_result

    

don_vi = {1:"đơn vị", 2:"chục", 3:"trăm", 4:"nghìn", 5:"chục nghìn",
         5:"trăm nghìn", 6:"triệu"}


def check_float_2(num):
    for index in range(len(num)):
        if num[index] != '0' and num[index] != '.':
            return num[:index + 1]
        

def get_true_format(num, digit):
    num_split = num.split(',')
    #print(num_split)
    num_before = num_split[0]
    #print(num_before)
    result = None
    #print(num_before)
    #print(digit)
    for index, each_digit in enumerate(num_before):
        if each_digit == digit:
            result = int(digit) * 10 ** (len(num_before) - index - 1)
            break
    if len(num_split) > 1:
        num_after = num_split[1]
        for index, each_digit in enumerate(num_after):
            if each_digit == digit:
                result = float(digit) * (float(1) / float(10 ** (index + 1)))
                break
    if result >= 1:
        result_txt = convert_float_value_to_text(str(result)).replace('mươi', 'chục')
        #print(result_txt)
    else:
        #print('qq')
        #print(check_float_2(str(result)).replace('.', ''))
        result = float(check_float_2(str(result)))
        result_txt = convert_int_value_to_text(check_float_2(str(result)).replace('.', '')[::-1])
        #print(result_txt)
        if result_txt != 'mười':
            result_txt = result_txt.split(' ')
            result_txt = result_txt[:1] + ['phần'] + result_txt[1:]
            result_txt = ' '.join(result_txt).replace('mươi', 'mười')
        else:
            result_txt = 'một phần mười'
        result_txt = [str(w2n(result_txt.split(' ')[0]))] + result_txt.split(' ')[1:]
        result_txt = ' '.join(result_txt)
    return result, result_txt


def adding_rule(question, choices):
    true_answer = {}
    
    data = {
        'id':1, 
        'question': preprocess(question, lowercase=True),  
        'choices': [preprocess(elem, lowercase=True) for elem in choices]
    }
    question = data['question'].lower()
    question = question.replace(".", "")
    split_question = question.split(' ')
    #print(question)
    
    
    try:
        if 'đọc' in question:
            split_question_1 = [convert_float_value_to_text(elem) for elem in split_question if check_float(elem)]
            split_question_2 = [convert_float_having_don_vi_to_text(elem) for elem in split_question if check_having_don_vi(elem)]
            #print(data['question'])
            if len(split_question_1) == 1:
                question_result = split_question_1[0]


            if len(split_question_2) == 1:
                question_result = split_question_2[0]
            #print(true_answer)
            save_special_choice = None
            for choice in data['choices']:
                true_choice = " ".join(choice.split(" ")[1:]).lower()
                #print(true_choice)
                if true_choice in question_result:
                    check = data['id'] not in true_answer
                    if data['id'] in true_answer:
                        recent_choice = " ".join(true_answer[data['id']].split(" ")[1:]).lower()
                        check = check or (recent_choice in true_choice)
                    if check:
                        true_answer[data['id']] = choice
            if data['id'] in true_answer:
                return true_answer[data['id']], True
    except:
        pass
    try:
        split_question_3 = [float(elem.replace(',', '.')) for elem in split_question if check_float(elem)]
        #print(split_question_3)
        #print("tổng" in question)
        if len(split_question_3) == 1:
            #print(question)
            if "tích" in question or "tổng" in question or "hiệu" in question or "thương" in question:
                save_special_choice = None
                for choice in data['choices']:
                    split_choice = np.asarray([float(elem.replace(',', '.')) for elem in choice.split(' ') if check_float(elem)])
                    print(split_choice)
                    #print(split_question_3[0])
                    #print(split_choice)
                    if len(split_choice) == 0:
                        save_special_choice = choice
                    if "tích" in question:
                        if np.prod(split_choice) == split_question_3[0]:
                            true_answer[data['id']] = choice
                            break
                    if "tổng" in question:
                        if np.sum(split_choice) == split_question_3[0]:
                            true_answer[data['id']] = choice
                            break
                    if "hiệu" in question: 
                        if abs(split_choice[0] - split_question_3[1]) == split_question_3[0]:
                            true_answer[data['id']] = choice
                            break
                    if "thương" in question: 
                        if split_choice[0]/split_choice[1] == split_question_3[0] or split_choice[1]/split_choice[0] == split_question_3[0]:
                            true_answer[data['id']] = choice
                            break
                if data['id'] in true_answer:
                    return true_answer[data['id']], True
    except:
        pass
    if True:
        if True:
            try:
                #print("\\frac" in question)
                #print(data['id'])
                if "\\frac" in question:
                    split_question_4 = [elem for elem in split_question if "\\frac" in elem]
                    split_question_5 = [elem for elem in split_question if check_float(elem)]
                    if len(split_question_4) == 1:
                        #print(question)
                        split_question_4 = split_question_4[0]
                        split_question_4 = split_question_4.split('}{')
                        #print(split_question_4)
                        if True:
                            mau_so = split_question_4[1].split('}')[0]
                            tu_so = split_question_4[0].split('{')[1]
                            question_result = float(tu_so) / float(mau_so)
                            if len(split_question_5) == 1:
                                question_result += float(split_question_5[0].replace(',', '.'))
                            #print(question_result)
                        else:
                            pass
                        save_special_choice = None
                        for choice in data['choices']:
                            #break
                            #print(choice)
                            true_choice = choice.split(" ")[1]
                            if check_float(true_choice):
                                choice_result = float(true_choice.replace(",", "."))
                                #print(choice_result)
                                if  choice_result == question_result:
                                    #print(choice_result)
                                    true_answer[data['id']] = choice
                                    break
                            else:
                                try:
                                    split_choice_2 = [elem for elem in choice.split(" ") if "\\frac" in elem]
                                    if len(split_choice_2) == 1:
                                        choice_result = convert_txt_phan_so_to_float(split_choice_2[0])
                                        if choice_result == question_result:
                                            #print(choice_result)
                                            true_answer[data['id']] = choice
                                            break
                                except:
                                    save_special_choice = choice
                        if data['id'] in true_answer:
                            return true_answer[data['id']], True
            except:
                pass
    split_question_6 = question.split("“")
    try:
        if len(split_question_6) > 1:
            split_question_6 = split_question_6[-1].split("”")[0].lower()
            #print(split_question_6)
            question_result = convert_text_to_float(split_question_6)
            save_special_choice = None
            for choice in data['choices']:
                #print(choice)
                true_choice = choice.split(" ")[1]
                #print(true_choice)
                if check_float(true_choice):
                    choice_result = float(true_choice.replace(",", "."))
                    #print(choice_result)
                    #print(choice_result)
                    if  choice_result == question_result:
                        #print(choice_result)
                        true_answer[data['id']] = choice
                        break
                if True:
                    if True:
                        if True:
                            split_choice_2 = [elem for elem in choice.split(" ") if "\\frac" in elem]
                            #print(split_choice_2)
                            if len(split_choice_2) == 1:
                                #print(split_choice_2[0])
                                #print(question_result)
                                choice_result = convert_txt_phan_so_to_float(split_choice_2[0])
                                #print(choice_result)
                                #print(choice_result)
                                if choice_result == question_result:
                                    true_answer[data['id']] = choice
                                    break
                else:
                    save_special_choice = choice
            if data['id'] in true_answer:
                return true_answer[data['id']], True
    except:
        pass
    try:
        if 'chữ số' in question and 'trong' in question:

            split_question_7 = [elem for elem in split_question if check_float(elem)]
            if len(split_question_7) == 2:

                save_special_choice = None
                digit = split_question_7[0]
                num = split_question_7[1]
                question_result_num, question_result = get_true_format(num, digit)
                #print(question_result_num, question_result)
                for choice in data['choices']: 
                    #print(choice)
                    true_choice = " ".join(choice.split(" ")[1:])


                    try:
                        if check_float(true_choice):
                            choice_result = float(true_choice.replace(",", "."))
                            #print(choice)
                            #print(choice_result)
                            if choice_result == question_result_num:
                                true_answer[data['id']] = choice
                                break
                        split_choice_2 = [elem for elem in choice.split(" ") if "\\frac" in elem]
                        if len(split_choice_2) == 1:
                            choice_result = convert_txt_phan_so_to_float(split_choice_2[0])
                            #print(choice_result)
                            if choice_result == question_result_num:
                                true_answer[data['id']] = choice
                                break
                    except:
                        save_special_choice = choice



                #print(question_result_num, question_result)
                for choice in data['choices']:
                    true_choice = " ".join(choice.split(" ")[1:]).lower()
                    if true_choice in question_result:
                        check = data['id'] not in true_answer
                        if data['id'] in true_answer:
                            recent_choice = " ".join(true_answer[data['id']].split(" ")[1:]).lower()
                            check = check or (recent_choice in true_choice)
                        if check:
                            true_answer[data['id']] = choice
                if data['id'] in true_answer:
                    return true_answer[data['id']], True

    except:
        pass
    return "", False


def adding_rule_back_up(question, choices):
    try:
        return adding_rule(question, choices) 
    except:
        return "", False
        
    