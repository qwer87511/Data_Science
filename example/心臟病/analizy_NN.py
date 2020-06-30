from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
import pandas as pd
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split as sk_split
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from os import listdir
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from keras.utils import plot_model 

def read_data(path):
    data=pd.read_csv(path)
    # data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    feature = data[data.columns[:-1]]
    target = data[data.columns[-1]]
    return feature, target


def build():
    model = Sequential()
    model.add(Dense(2048, input_dim=13))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model

def train_plot(path):
    f = pd.read_csv(path, index_col = 0)
    # print(f)
    epoch = f.index
    t_acc = f["acc"]
    t_loss = f['loss']
    v_acc = f["val_acc"]
    v_loss = f['val_loss']
    plt.plot(epoch,t_loss)
    plt.cla()
    # plt.show()

def visualization_roc(tpr, fpr,path):
    plt.yticks(fontsize = 2)
    plt.plot(fpr, tpr,c =" .3")
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig("model_save/{}roc.png".format(path))
    plt.cla()
    # plt.show()

def find_hdf5(x):
    if 'hdf5' in x:
        return x

def analizy(model_path, x, y):
    dic = {}
    paths = list(map(lambda x : find_hdf5(x),listdir(model_path)))
    for path in paths:
        try:
            temp = []
            model = load_model(model_path+'/'+path)
            pre = list(map(lambda x : np.argmax(x), model.predict(x)))
            tn, fp, fn, tp = confusion_matrix(y,pre).ravel()
            fpr, tpr, thresholds = roc_curve(y,pre)
            temp.append(tp/(tp+fn))
            visualization_roc(tpr,fpr,path)
            Auc = auc(fpr, tpr)
            temp.append(Auc)
            dic[path]=temp
            plot_model(model, to_file='model_save/' + path + '_model.png', show_shapes=True, show_layer_names=True)
        except:
            pass
    df = pd.DataFrame.from_dict(dic,orient='index')
    df = df.reset_index()
    df.columns = ['model','sensitivy','auc']
    df.to_csv(model_path+'/auc_sen_data.csv')


def train(model, x, y):
    record_path = 'model_save/training.csv'

    train_x, test_x, train_y, test_y = sk_split(x, y, test_size=0.3, random_state=10) # 以隨機的方式資料分割 並給隨機種子固定隨機模式
    train_y = keras.utils.to_categorical(train_y, 2)
    test_y = keras.utils.to_categorical(test_y, 2)
    adam = Adam(lr=0.05, decay=3e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
    # 將每次訓練的結果已CSV檔形式儲存起來 => 可以用來視覺化訓練情況
    csv_logger = CSVLogger(record_path)
    filepath='model_save'
    checkpointer = ModelCheckpoint(filepath=filepath+'/weights-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True,period=10)
    # model.fit(train_x, train_y, batch_size=20, epochs=500, validation_data=(test_x,test_y), verbose=1, callbacks=[csv_logger, checkpointer])
    train_plot(record_path)
    analizy(filepath, x, y)
    

def main():
    model = build()
    # ann_viz(model,title="")
    x,y = read_data("heart.csv")
    train(model, x, y)

main()