"""providing load_mnist function"""
import os
import os.path
import urllib.request
import gzip
import pickle
import numpy as np

URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

TRAIN_NUM = 60000
TEST_NUM = 10000
img_dim = (1, 28, 28)
IMG_SIZE = 784

def _download(file_name):
    """ file_name 이름의 데이터를 urllib requset 메소드를 통해
    현재 디렉토리의 똑같은 폴더를 추가해서 다운로드한다.
    """
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(URL_BASE + file_name, file_path)
    print("Done")

def download_mnist():
    """ key_file 의 value에 파일 이름들이 저장되어있고, 
    _download 함수를 이용해 디렉토리를 만들고 다운로드한다.
    """
    for v in key_file.values():
        _download(v)

def _load_label(file_name):
    """gzip 파일을 메모리에 올리고 
    Numpy의 frombuffer 메소드를 이용해서 넘파이 배열로 변환한다."""
    file_path = dataset_dir + "/" + file_name

    print("Converting" + file_name + "to Numpy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset= 8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting" + file_name + "to Numpy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset= 16)

    data = data.reshape(-1, IMG_SIZE)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def init_mnist():
    """초기 실행 함수"""
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ... ")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

    print('Done')

def _change_one_hot_label(x):
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'],dataset['train_label']),(dataset['test_img'],dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
