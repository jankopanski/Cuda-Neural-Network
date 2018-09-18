#!/usr/bin/env python3

from sys import argv, stdout
from os import walk
from os.path import basename, splitext, join
from PIL import Image
from numpy import asarray, append
from csv import writer
from random import shuffle


IMAGE_SIZE = 64

rootdir = argv[1]
output_path = argv[2]
output = []

diriter = walk(rootdir)
next(diriter)
for (dirpath, _, filenames) in diriter:
    target = int(basename(dirpath))
    metadata_file_index = next(i for (i, f) in enumerate(filenames) if splitext(f)[1] == '.csv')
    # metadata_file = filenames[metadata_file_index]
    del filenames[metadata_file_index]
    for image_file in filenames:
        image_path = join(dirpath, image_file)
        image = Image.open(image_path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("L")
        arr = asarray(image).flatten()
        output.append(append(arr, target))

shuffle(output)
output_file = stdout if output_path == '-' else open(output_path, 'w')
output_writer = writer(output_file, delimiter=' ')
output_writer.writerow([len(output)])
output_writer.writerows(output)
