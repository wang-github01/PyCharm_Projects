import os
# import cv2

def data_loader(datadir, batch_size=10, mode='train'):
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # reandom.shuffile(filenames)
            barch_imgs = []
            batch_labels = []
            for name in filenames:
                filepath = os.path.join(datadir, name)
                img = cv2.im