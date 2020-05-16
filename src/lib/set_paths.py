import os
import sys


def get_paths(in_type):

    # path to file : paths.cfg
    p = 'paths.cfg'

    file = open(p, 'r')
    lines = file.readlines()

    all_paths = list()
    req = list()
    req_paths = list()
    home = ''

    for line in lines:
        if line[0] not in ('#', ' ', '\n'):
            all_paths.append(line)

    flag = True
    for i in all_paths:

        if i[0] == 'M':
            dir_name = i.split('=')[1].strip()
            ind = os.getcwd().split('\\').index(dir_name)
            a = (os.getcwd().split('\\')[0:ind+1])
            home = home.join(x + '\\' for x in a)
            req_paths.append(home)

        elif i[0] == 'I':
            i = i[2:].split('=')[1].strip()
            if not os.path.exists(home + i):
                print('Path Not found : ' + i)
                flag = False
        else:
            continue

    if in_type == 'feature_extractor':
        req = ['path_dataset', 'path_train_set', 'path_test_set', 'path_extracted_train_features',
               'path_extracted_test_features']

    if in_type == 'embedding_loader':
        req = ['path_glove_txt']

    if in_type == 'model_trainer':
        req = ['path_desc', 'path_train_set', 'path_extracted_train_features']

    if in_type == 'generate_descriptions':
        req = ['path_desc', 'path_tokens']

    req_paths.append([x.split('=')[1].strip() for x in all_paths if x.split()[0].split('.')[1] in req])

    return req_paths, flag


def run_path_check(use_type):

    use_paths, status = get_paths(use_type)
    print("Errors Detected in paths!! Terminating Program" if not status else '')

    if not status:
        sys.exit()

    return use_paths

