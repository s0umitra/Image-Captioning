import os
from src.lib.descrip_lib import load_descriptions, clean_descriptions, save_descriptions
from src.lib.libic import init, get_program_name


def initialize():

    # get program name
    caller = get_program_name()

    # initiate
    paths = init(caller)

    # set home path
    path_home = paths[0]
    os.chdir(path_home)

    # set paths
    path_tokens, path_descriptions = paths[1]

    # load descriptions
    file_name = open(path_tokens, 'r')
    desc = file_name.read()

    return desc, path_descriptions


if __name__ == "__main__":

    raw_descriptions, save_path = initialize()

    # parse descriptions
    all_descriptions = load_descriptions(raw_descriptions)
    print('Images        : %d ' % len(all_descriptions))

    # clean descriptions
    all_descriptions = clean_descriptions(all_descriptions)

    # save to file
    count = save_descriptions(all_descriptions, save_path)

    print('Descriptions  :', count)
    print('File saved to :', save_path)
