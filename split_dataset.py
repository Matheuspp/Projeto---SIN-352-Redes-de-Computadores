import pathlib
from shutil import copy
from sklearn.model_selection import train_test_split

def read_images(path):
    files = sorted(pathlib.Path(path).glob('*.png'))
    return files

def split(files):
    train, val = train_test_split(files, test_size=0.4, random_state=10, shuffle=True)
    val, test  = train_test_split(val, test_size=0.5, random_state=10, shuffle=True)
    return train, val, test

def copy_imgs(imgs, folder, s_set):
    for img in imgs:
        pathlib.Path(f'{str(img.parent.parent)}/{s_set}/{folder}').mkdir(exist_ok=True, parents=True)
        print(img)
        dest = f'{str(img.parent.parent)}/{s_set}/{folder}/{img.stem}.png'
        copy(img, dest)


if __name__ in '__main__':
    path = 'img_dataset'

    folders = sorted(pathlib.Path(path).glob('*'))
    print(folders)
    for folder in folders:
        files = read_images(f'{path}/{str(folder.stem)}/')
        train, val, test = split(files)
        copy_imgs(train, str(folder.stem), 'train')
        copy_imgs(val, str(folder.stem), 'val')
        copy_imgs(test, str(folder.stem), 'test')

    print('Done!')


