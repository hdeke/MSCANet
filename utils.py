import wfdb
import pywt
import seaborn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split


def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def get_data_set(number, X_data, Y_data):
    ecgClassSet = ['N', 'S', 'V', 'F', 'Q']

    # load the ecg data record
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)


    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end


    while i < j:
        try:

            label = None
            if Rclass[i] in ['N', 'L', 'R', 'e', 'j']:
                label = ecgClassSet.index('N')
            elif Rclass[i] in ['A', 'a', 'J', 'S']:
                label = ecgClassSet.index('S')
            elif Rclass[i] in ['V', 'E']:
                label = ecgClassSet.index('V')
            elif Rclass[i] == 'F':
                label = ecgClassSet.index('F')
            elif Rclass[i] in ['/', 'Q', 'f']:
                label = ecgClassSet.index('Q')
            if label is not None:

                x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
                X_data.append(x_train)
                Y_data.append(label)
            i += 1
        except ValueError:
            i += 1
    return


def load_data(ratio, random_seed):
    numberSet = ['100', '101', '103', '105', '106', '107', '108',
                 '109', '111', '112', '113', '114', '115','116', '117', '118',
                 '119', '121', '122', '123', '124', '200', '201', '202', '203',
                 '205', '207', '208', '209', '210', '212', '213', '214', '215',
                 '217', '219', '220', '221', '222', '223', '228', '230','231',
                 '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        get_data_set(n, dataSet, lableSet)
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1,1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)
    X_train, X_test, y_train, y_test = train_test_split(dataSet, lableSet, test_size=ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test

def plot_heat_map(y_test, y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png',dpi=600)
    plt.show()

def plot_history_tf(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png',dpi=600)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png',dpi=600)
    plt.show()