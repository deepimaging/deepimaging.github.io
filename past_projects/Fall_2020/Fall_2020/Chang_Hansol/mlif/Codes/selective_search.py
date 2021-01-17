# -*- coding: utf-8 -*-
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk
import cv2


def _generate_segments(im_orig, scale, sigma, min_size):

    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):

    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):

    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_colour_hist(img):


    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        c = img[:, colour_channel]
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])
    hist = hist / len(img)

    return hist


def _calc_texture_gradient(img):

    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img):

    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        fd = img[:, colour_channel]

        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])


    hist = hist / len(img)

    return hist


def _extract_regions(img):
    R = {}


    hsv = skimage.color.rgb2hsv(img[:, :, :3])


    for y, i in enumerate(img):

        for x, (r, g, b, l) in enumerate(i):


            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}


            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    tex_grad = _calc_texture_gradient(img)

    for k, v in R.items():

        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R


def _extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = regions.items()
    r = [elm for elm in R]
    R = r
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
                          r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
                          r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50):

    assert im_orig.shape[2] == 3, "3ch image is expected"


    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)

    neighbours = _extract_neighbours(R)


    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    while S != {}:

        i, j = sorted(list(S.items()), key=lambda a: a[1])[-1][0]

        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        for k in key_to_delete:
            del S[k]

        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions


def iou(axes, nodule):

    vertice1 = [axes[0], axes[1], axes[0] + axes[2], axes[1] + axes[3]]
    vertice2 = [nodule[0], nodule[1], nodule[0] + nodule[2], nodule[1] + nodule[3]]
    lu = numpy.maximum(vertice1[0:2], vertice2[0:2])
    rd = numpy.minimum(vertice1[2:], vertice2[2:])
    intersection = numpy.maximum(0.0, rd - lu)
    inter_square = intersection[0] * intersection[1]
    square1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    square2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])
    union_square = numpy.maximum(square1 + square2 - inter_square, 1e-10)
    return numpy.clip(inter_square / union_square, 0.0, 1.0)


def store_info(img_rgb, axes, nodule):
    info = {
        "array": img_rgb[y:y + h, x:x + w, :],
        "axes": [x, y, w, h],
        "label": False}
    if iou(axes, nodule)[0] > 0.3:
        info["label"] = True
    return info


if __name__ == '__main__':
    img_path = "./1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
    img = skimage.io.imread(img_path)
    example_img = cv2.normalize(img[68], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_rgb = cv2.cvtColor(example_img, cv2.COLOR_GRAY2BGR)

    img_lbl, regions = selective_search(
        img_rgb, sigma=0.5, scale=200, min_size=20)

    candidates = set()
    for r in regions:

        if r['rect'] in candidates:
            continue

        if r['size'] < 2000:
            continue

        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])


    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img_rgb)
    for x, y, w, h in candidates:
        print(x, y, w, h)


        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()
