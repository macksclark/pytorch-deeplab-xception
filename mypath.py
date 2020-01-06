import os
from torchvision.datasets import voc

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            base_dir = os.path.join(os.getcwd(), "data", "pascal")
            trainval_dir = os.path.join(base_dir, "VOCdevkit", "VOC2012")
            if not os.path.isdir(trainval_dir):
                # Need to download the VOC dataset to this folder
                voc.VOCSegmentation(base_dir, year='2012', image_set='trainval', download=True)
            return trainval_dir
        elif dataset == 'sbd':
            pass
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
