# gray scale 1 channel image
import numpy as np
import pydicom
import pandas as pd
import os


class ImagesPixelMean():
    def __init__(self, datasetdir, csvpath):
        self.csvpath = csvpath
        self.datasetdir = datasetdir

    def getMeanImage(self):
        annotations = pd.read_csv(self.csvpath)
        annotations.columns = ['ImageId', 'ImageEncoding']
        image_ids = annotations.iloc[:, 0].values
        rles = annotations.iloc[:, 1].values
        images = []
        for row in annotations.itertuples():
            id = row.ImageId
            encoding = row.ImageEncoding
            image_path = os.path.join(self.datasetdir, id + ".dcm")
            pyimage = pydicom.dcmread(image_path)
            height = pyimage.Rows
            width = pyimage.Columns
            image = pyimage.pixel_array   ## pixel array is shape (2,2)
            images.append(image)      # creates (n, 1024,1024)

        images = np.array(images)
        meanimage = np.mean(images, axis=0)   #take mean across axis = 0 returns ( 1024,1024)
        pixelmean = np.mean(meanimage)
        return pixelmean, meanimage


if __name__ == '__main__':

    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate pixel mean for  training set')

    parser.add_argument('--datasetdir', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the dcm files')
    parser.add_argument('--csvpath', required=True,
                        metavar="/path/to/csv file",
                        help="Path to csv file")

    args = parser.parse_args()

    # Validate arguments

    assert args.datasetdir, "Argument --dataset is required"
    assert args.csvpath , "Argument --csvpath is required"

    imagepixels = ImagesPixelMean(args.datasetdir, args.csvpath)
    pixel_mean , mean_image = imagepixels.getMeanImage()
    print(pixel_mean)
    print(mean_image.shape)





