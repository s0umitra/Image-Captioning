import string

from src.lib.libic import desc_loader


def load_set(filename):

    doc = desc_loader(filename)
    dataset = list()

    for line in doc.split('\n'):

        if len(line) < 1:
            continue

        i_name = line.split('.')[0]
        dataset.append(i_name)

    return set(dataset)


def load_clean_descriptions(filename, dataset):

    doc = desc_loader(filename)
    descriptions = dict()

    for line in doc.split('\n'):

        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]

        if image_id in dataset:

            if image_id not in descriptions:
                descriptions[image_id] = list()

            desc = '<start> ' + ' '.join(image_desc) + ' <end>'
            descriptions[image_id].append(desc)

    return descriptions


def load_descriptions(file_name):

    desc_mappings = dict()

    for line in file_name.split('\n'):

        tokens = line.split()

        if len(line) < 2:
            continue

        image_name, image_desc = tokens[0], tokens[1:]
        image_name = image_name.split('.')[0]
        image_desc = ' '.join(image_desc)

        if image_name not in desc_mappings:
            desc_mappings[image_name] = list()

        desc_mappings[image_name].append(image_desc)

    return desc_mappings


def clean_descriptions(descriptions):

    table = str.maketrans('', '', string.punctuation)

    for key, desc_list in descriptions.items():

        for i in range(len(desc_list)):

            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

    return descriptions


def save_descriptions(descriptions, file_name):

    count = 0
    lines = list()

    for key, desc_list in descriptions.items():

        for desc in desc_list:

            lines.append(key + ' ' + desc)
            count += 1

    data = '\n'.join(lines)
    file = open(file_name, 'w')
    file.write(data)
    file.close()

    return count


def to_lines(descriptions):

    all_desc = list()

    for key in descriptions.keys():

        [all_desc.append(d) for d in descriptions[key]]

    return all_desc


def get_max_length(descriptions):

    lines = to_lines(descriptions)

    return max(len(d.split()) for d in lines)
