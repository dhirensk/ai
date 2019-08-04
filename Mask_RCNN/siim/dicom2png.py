import os
import pydicom
import glob
from PIL import Image

inputdir = './samples/'
outdir = './out/'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
#glob.glob(inputdir + './*.dcm')
for f in test_list:
    ds = pydicom.read_file( inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface
    img_mem.save(outdir + f.replace('.dcm','.png'), "PNG")