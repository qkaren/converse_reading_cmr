import re
import os
from itertools import chain


def filter_text(text):
    #https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
    text = re.sub(r'[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?', '<NUM>', text)
    text = re.sub("[\(].*?[\)]", "", text)    # [\[\(].*?[\]\)]
    text = text.split()
    new_text = []
    for x in text:
        if 'www' in x or 'http' in x:
            continue
        new_text.append(x)
    return ' '.join(new_text)


def filter_query(text, max_len=100):
    'removing the history of query'
    if 'EOS' in text:
        text = text[text.rindex('EOS')+4:]
    text = filter_text(text)
    if text.startswith('til'):
        text = re.sub('til ', '', text, 1)
    if text.startswith('...'):
        text = re.sub('...', '<TRNC>', text, 1)
    if len(text.split()) > 50 and text.endswith('...'):
        text = text[:-3] + '<TRNC>'
    text = text.strip().split()
    if len(text) > max_len:
        text = ['<TRNC>'] + text[-max_len:]
    return ' '.join(text)


def filter_fact(text):
    text = filter_text(text)
    text = re.sub('- wikipedia ', '', text, 1)
    text = re.sub(' \[ edit \]', '', text, 1)
    text = re.sub('<h2> navigation menu </h2>', '', text)
    return text


def filter_resp(text):
    text = re.sub('\[|\]', '', text)
    return filter_text(text)
    # anything else?


def no_label(fact):
    include_labels = [
        '<title>', '<anchor>', '<p>', '<h1>', '<h2>', '<h3>', '<h4>',
        '</title>', '</anchor>', '</p>', '</h1>', '</h2>', '</h3>', '</h4>']
    for il in include_labels:
        if il in fact:
            return False
    return True


def write_files(output_path, data):
    with open(output_path + '.full', 'w', encoding='utf8') as fw_full:
        with open(output_path + '.query', 'w', encoding='utf8') as fw_query:
            with open(output_path + '.response', 'w', encoding='utf8') as fw_response:
                with open(output_path + '.fact', 'w', encoding='utf8') as fw_fact:
                    for line in data:
                        fw_query.write(' '.join(line['query']) + '\n')
                        fw_response.write(' '.join(line['response']) + '\n')
                        fw_fact.write(' '.join(line['fact']) + '\n')
                        fw_full.write(line['raw'])


def load_facts(fact_path):
    fact_dict = {}
    anchor_idx = 0
    anc_flag = False
    with open(fact_path, 'r', encoding='utf8') as fin:
        for i, line in enumerate(fin):
            parts = line.strip().split('\t')
            if len(parts) != 5:
                print('[Warning] loss fact #parts !=5, line %d in %s'
                      % (i, fact_path))
                continue
            fact_id = parts[2].strip()
            fact = filter_fact(parts[-1].strip()).split()

            if no_label(fact):
                continue

            if len(fact) > 100:
                fact = fact[:100] + ['<TRNC>']

            if fact_id in fact_dict:
                anchor_idx += 1
                fact_dict[fact_id]['facts'].append(fact)
                if '<anchor>' in fact:
                    anc_flag = True
                    fact_dict[fact_id]['anchor_idx'] = anchor_idx
                    fact_dict[fact_id]['anchor_label'] = fact[0]
            else:
                anchor_idx = 0
                anc_flag = False
                fact_dict[fact_id] = {}
                fact_dict[fact_id]['anchor_idx'] = anchor_idx
                fact_dict[fact_id]['anchor_label'] = ''
                fact_dict[fact_id]['facts'] = [fact]
            fact_dict[fact_id]['anchor_status'] = anc_flag
    return fact_dict


def combine_fact(fact_dict, anc_type='section', fact_len=12, just_anc=False):
    anc_end_idx = 0
    ret_fact_dict = {}
    for fact_id in fact_dict.keys():
        facts = fact_dict[fact_id]['facts']
        anc_idx = fact_dict[fact_id]['anchor_idx']
        anc_label = fact_dict[fact_id]['anchor_label']
        anc_flag = fact_dict[fact_id]['anchor_status']

        if just_anc and not anc_flag:
            continue

        if not anc_flag:
            facts = facts[:fact_len]
        else:
            if anc_type == 'sentence':
                anc_end_idx = anc_idx + 2
            elif anc_type == 'section':
                for anc_i, anc_text in enumerate(facts[anc_idx + 1:]):
                    if anc_label == anc_text[0]:  # <p>
                        anc_end_idx = anc_idx + anc_i + 1
                        break
            else:
                print('anchor type error')
                exit()
            facts = facts[:2] + facts[anc_idx:anc_end_idx]

        ret_fact_dict[fact_id] = fact_dict[fact_id]
        facts = ' '.join(list(chain(*facts)))
        ret_fact_dict[fact_id]['facts'] = facts
    return ret_fact_dict


def load_conv(conv_path, fact_dict, is_train=True, min_que=5):
    conv_fact_list = []
    count_min = 0
    count_dup = 0
    count_short_fact = 0
    hash_set = set()
    with open(conv_path, 'r', encoding='utf8') as fin:
        for i, line in enumerate(fin):
            parts = line.strip().split('\t')
            if len(parts) != 7:
                print('[Warning] loss convos #parts != 7, line %d in %s'
                      % (i, conv_path))
                continue

            hash_id = parts[0].strip()
            if hash_id in hash_set:
                count_dup += 1
                continue
            else:
                hash_set.add(hash_id)

            conv_id = parts[2].strip()
            query = filter_query(parts[-2].strip()).split()
            response = filter_resp(parts[-1].strip()).split()

            if conv_id in fact_dict:
                facts = fact_dict[conv_id]['facts']
                facts = filter_text(facts)
            else:
                continue

            facts = facts.split()

            if is_train:
                if len(response) < 5:
                    count_min += 1
                    continue

                if len(query) <= 1 or \
                    len(facts) <= 1:
                    continue
            else:
                if len(query) == 0:
                    query = ['UNK']
                if len(facts) == 0:
                    facts = ['UNK']

            if len(facts) < 10:
                count_short_fact += 1

            if len(facts) > 500:
                facts = facts[:500] + ['<TRNC>']

            if len(response) > 30:
                response = response[:30]

            conv_fact_dict = {
                'query': query,
                'response': response,
                'fact': facts,
                'conv_id': conv_id,
                'raw': line,
                'hash_id': hash_id
            }
            conv_fact_list.append(conv_fact_dict)
    return conv_fact_list, count_min, count_dup, count_short_fact


def combine_files(files_path, anc_type='section', fact_len=12, just_anc=False, is_train=False):
    file_name = set()
    data_list = []
    count_mins = 0
    count_dups = 0
    count_short_facts = 0
    for file in os.listdir(files_path):
        file_name.add(file[:file.find('.')])
    print(file_name)

    for file in file_name:
        init_path = os.path.join(files_path, file)
        fact_dict = load_facts(init_path + '.facts.txt')
        comb_fact = combine_fact(fact_dict, anc_type, fact_len, just_anc)
        conv_fact, count_min, count_dup,count_short_fact = load_conv(init_path + '.convos.txt', comb_fact, is_train)
        data_list += conv_fact
        count_mins += count_min
        count_dups += count_dup
        count_short_facts += count_short_fact
    print(len(data_list))
    print('have discard the short response {} times'.format(count_mins))
    print('the percentage of discarding is {0:.2f}%'.format(count_mins / len(data_list) * 100))
    print('there are {} hash value are the same'.format(count_dups))
    print('the percentage of duplicates is {0:.2f}%'.format(count_dups / len(data_list) * 100))
    print('there are {} count_short_facts'.format(count_short_facts))
    print('the percentage of count_short_facts is {0:.2f}%'.format(count_short_facts / len(data_list) * 100))
    return data_list
