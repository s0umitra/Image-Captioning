import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster


def get_avg(inp):
    size = len(inp)
    tot = 0
    for i in inp:
        tot += i

    return tot / size


def get_dominant_color(image):

    clusters = 5

    im = Image.open(image)
    im = im.resize((150, 150))
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, clusters)
    vec, dist = scipy.cluster.vq.vq(ar, codes)
    counts, bins = scipy.histogram(vec, len(codes))

    index_max = scipy.argmax(counts)
    peak_color = codes[index_max]

    return peak_color


def process_text(text):

    pro_txt = ''
    word = ""

    for i in range(len(text)):
        word += text[i]

        if i % 34 == 0 and i != 0:
            pro_txt += '\n'

        if text[i] == ' ':
            pro_txt += word
            word = ''

    if word != '':
        pro_txt += word

    return pro_txt


def draw(image_name, text):
    out_path = 'outputs\\' + image_name.split('\\')[-1]

    img = plt.imread(image_name)

    fig, ax = plt.subplots()
    plt.imshow(img)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    txt = process_text(text)
    lines = txt.split('\n')

    max_val = 0
    for line in lines:
        if max_val < len(line):
            max_val = len(line)

    plot_shape = plt.rcParams["figure.figsize"]
    plot_width = plot_shape[0]

    fs = int((plot_width / max_val) * 100)

    if fs not in range(10, 21):
        fs = 16

    b_color = get_dominant_color(image_name)
    b_color = [x / 255.0 for x in b_color]

    f_color = get_avg(b_color)

    if f_color > 0.5:
        f_color = 'black'
    else:
        f_color = 'white'

    plt.xlabel(txt,
               fontsize=fs, style='italic', color=f_color,
               bbox=dict(facecolor=b_color, edgecolor='white', alpha=0.9, boxstyle='round'),
               labelpad=9)

    plt.savefig(out_path,  bbox_inches="tight")
