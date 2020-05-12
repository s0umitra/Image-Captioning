import os


def get_paths(in_type):

    # path to file : paths.cfg
    p = '..\\..\\paths.cfg'
    file = open(p, 'r')
    lines = file.readlines()

    all_paths = list()
    req_paths = list()

    for line in lines:
        if line[0] not in ('#', ' ', '\n'):
            all_paths.append(line)

    if in_type == 'feature_extractor':
        req = ['path_dataset', 'path_train_set']
        req_paths = [x.split('=')[1].strip() for x in all_paths if x.split()[0] in req]

    return req_paths


def path_verify(paths):

    flag = True
    for i in paths:
        if os.path.exists(i):
            print('Verified  : ' + i)
        else:
            print('Not found : ' + i)
            flag = False

    return flag


def run_path_check(use_type):

    use_paths = get_paths(use_type)
    status = path_verify(use_paths)
    print('All Good' if status else 'Errors Detected')

    return use_paths, status

