import jieba
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import os
import random
from sklearn.metrics import roc_curve, auc
import time
def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # `pip install tensorflow-determinism` first,使用与tf>2.1

# # LSTM_Variable
global lr_value
global lstm_random_seed
global lstm_labels_name
global lstm_test_set_size
global EPOCH
global BATCH_SIZE
# # LSTM_Setting
lr_value = 0.001  # 學習率越高，模型學習效果越好，越小則學習效果降低
lstm_random_seed = 101  # 固定隨機種子數值，使輸出可以固定
lstm_test_set_size = 0.3  # 測試集占比大小
lstm_labels_name = '是否具有創意'  # 原始文本中，標籤的表頭名稱
EPOCH = 1
BATCH_SIZE = 2
# 移除標點符號 & 無意義字
def remove_stop_words(file_name,seg_list):
    with open(file_name,'r', encoding='UTF-8') as f:
        stop_words = f.readlines()
    stop_words = [stop_word.rstrip() for stop_word in stop_words]

    new_list = []
    for seg in seg_list:
        seg = seg.replace('.','')
        if seg not in stop_words and seg.isdigit() is False:
            if seg.isalpha() == True and len(seg) == 1:
                continue
            elif seg == ' ' or seg.encode('UTF-8').isalpha() == True or seg == '　':
                continue
            elif bool(re.search('[a-z]', seg)) == True:
                continue
            else:
                new_list.append(seg)
    return new_list
def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def train_lstm():
    seed_tensorflow(555)  # 555
    filename = 'result.csv'
    print('模型運算中...')
    concat_df = pd.read_csv('temp_data\\' + filename, header=None, low_memory=False) # 讀取已進行tf-idf轉換，且將標籤結合的矩陣的檔案

    concat_df.iloc[0, 0] = '字詞特徵'
    new_col = concat_df.iloc[0, :]
    print('總字詞特徵數量: ' + str(len(new_col))) # 計算出所有字詞特徵的數量

    concat_df.columns = new_col
    concat_df = concat_df.iloc[1:, 1:] # 去除表頭和列名

    X = concat_df.drop(labels=[lstm_labels_name], axis=1).values  # 移除'是否具有創意'column，並取得剩下欄位資料
    y = concat_df[lstm_labels_name].values #　取得所有列的標籤值
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=lstm_test_set_size, random_state=lstm_random_seed) # 將資料依照test_set_size 比例切分(此參數代表測試集占所有資料比)

    # 將資料numpy array 轉換成float 型態，得以和tensor 型態相容
    X_train = X_test.astype(np.float32)
    y_train = y_test.astype(np.float32)
    tensor_X = tf.convert_to_tensor(X_train)
    tensor_y = tf.convert_to_tensor(y_train)

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    tensor_X_test = tf.convert_to_tensor(X_test)
    tensor_y_test = tf.convert_to_tensor(y_test)

    # 建模
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(activation='relu', filters=64, kernel_size=3, input_shape=(tensor_X.shape[1], 1)),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Conv1D(activation='relu', filters=32, kernel_size=3),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid', name='Dense_3')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=lr_value)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.TruePositives(name = 'TP'),
                           tf.keras.metrics.TrueNegatives(name = 'TN'),
                           tf.keras.metrics.FalsePositives(name = 'FP'),
                           tf.keras.metrics.FalseNegatives(name = 'FN'),
                           f1])

    col = ['Epoch '+str(i+1) for i in range(EPOCH)]
    tensor_X = np.expand_dims(tensor_X, axis=-1)  # 將 TF-IDF 特徵轉換為 3D 張量，為了要符合conv的input需求，須將2D轉3D
    X_test_tfidf = np.expand_dims(tensor_X_test, axis=-1) # 第三個維度的值皆設為1
    # 訓練開始時間
    start_time = time.time()
    history = model.fit(tensor_X, y_train,
              batch_size = BATCH_SIZE,
              epochs = EPOCH)
    # 訓練結束時間
    end_time = time.time()

    # 計算訓練時間（秒）
    training_time = end_time - start_time
    X_test_tfidf = np.expand_dims(X_test_tfidf, axis=-1)
    # 測試開始時間
    test_start_time = time.time()

    history_pd = pd.DataFrame.from_dict(history.history, orient='index')
    history_pd.columns = col
    history_pd.to_excel('tfidf_lstm_output.xlsx')

    print(history.history['accuracy']) # [0.0, 0.0, 0.0]
    print(history.history['TP']) # [0.0, 0.0, 0.0]
    print(history.history['FP']) # [0.0, 0.0, 0.0]
    print(history.history['TN']) # [0.0, 0.0, 0.0]
    print(history.history['FN']) # [0.0, 0.0, 0.0]


    predictions = model.predict(X_test_tfidf)
    # 測試結束時間
    test_end_time = time.time()

    # 計算測試時間（秒）
    testing_time = test_end_time - test_start_time
    # 取得accuracy、precision、recall、f1等數據
    acc = history.history['accuracy']  # Accuracy
    precision = history.history['precision']  # Precision
    recall = history.history['recall']  # Recall
    f1_score = history.history['f1']  # F1
    avg_acc = 0
    avg_pre = 0
    avg_reca = 0
    avg_f1 = 0
    for number in range(len(acc)):
        avg_acc = avg_acc + acc[number]
    avg_acc = float(avg_acc/len(acc))

    for number in range(len(precision)):
        avg_pre = avg_pre + precision[number]
    avg_pre = float(avg_pre/len(precision))

    for number in range(len(recall)):
        avg_reca = avg_reca + recall[number]
    avg_reca = float(avg_reca/len(recall))

    for number in range(len(f1_score)):
        avg_f1 = avg_f1 + f1_score[number]
    avg_f1 = float(avg_f1/len(f1_score))

    dense3_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_3').output)
    # 計算AUC
    dense3_output = dense3_layer_model.predict(tensor_X)

    fpr_3, tpr_3, thresholds_keras = roc_curve(y_train, dense3_output.ravel())
    roc_auc_3 = auc(fpr_3, tpr_3)

    # time.sleep(99)
    with open('./tfidf_Report.txt','a') as f:
        f.writelines("LSTM model train report:")
        f.write("訓練時間：{} 秒 \n".format(training_time))
        f.write("測試時間：{} 秒 \n".format(testing_time))
        f.writelines("AUC: {:.4f} \n".format(roc_auc_3))
        f.writelines("Accuracy: {:.4f} \n".format(avg_acc))
        f.writelines("Precision: {:.4f} \n".format(avg_pre))
        f.writelines("Recall: {:.4f} \n".format(avg_reca))
        f.writelines("F1: {:.4f} \n".format(avg_f1))
        f.write("\n")
    print("LSTM模型運算完成")
    model.save('lstm_tfidf.h5')
    return fpr_3, tpr_3, roc_auc_3