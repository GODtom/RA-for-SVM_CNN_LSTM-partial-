import jieba
import pandas as pd
import re
from gensim.models import word2vec
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import os
import random
import time
from sklearn.metrics import roc_curve, auc
# from UtilWordEmbedding import MeanEmbeddingVectorizer

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
BATCH_SIZE = 8
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

def word2vec_chang(filename):
    import csv
    import numpy as np
    from gensim.models import Word2Vec

    def adjust_lists(lists, size):
        adjusted_lists = []

        for lst in lists:
            adjusted_lst = lst[:size] + [0] * (size - len(lst)) if len(lst) < size else lst[:size]
            adjusted_arr = np.array(adjusted_lst)
            if(len(adjusted_lists) == size):
                return adjusted_lists
            adjusted_lists.append(adjusted_arr)

        while len(adjusted_lists) < size:
            adjusted_lists.append(np.zeros_like(lists[0]))

        return adjusted_lists

    def readfile(file_name):
        # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
        content_list = []
        label_list =[]
        with open(file_name, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # 去除表頭
            for row in csvreader:
                temp = []
                temp.append(row[0])
                label_list.append(row[1])
                content_list.append(temp)
            return content_list,label_list

    content_list,label_list = readfile(filename)
    clean_word_list = []
    total_number = 0

    for number in range(len(content_list)):
        temp = jieba.lcut(content_list[number][0])
        temp = remove_stop_words('remove_format.txt', temp)
        total_number = total_number + len(temp)
        clean_word_list.append(temp)

    word_pad_list = []
    word2vec_model = Word2Vec.load('word2vec.model')

    for num in range(len(clean_word_list)):
        word_vectors = [word2vec_model.wv[word] for word in clean_word_list[num] if word in word2vec_model.wv] # 將這篇文章的字詞轉成200維的數字矩陣

        # 將每篇文章的詞向量序列轉換為相同的長度，長度為max_length
        max_length = 200
        word_vectors = adjust_lists(word_vectors,max_length)

        word_pad_list.append(word_vectors)

    return word_pad_list
def train_lstm(filename):
    seed_tensorflow(555)  # 555
    # 讀取文本資料
    data = pd.read_csv(filename)

    # 將每一筆資料的content欄位轉換成由單詞構成的列表
    sentences = [doc.split() for doc in data['Content']]

    # 設定Word2Vec模型的參數
    seed = 666
    sg = 0
    window_size = 10
    vector_size = 100
    min_count = 1
    workers = 8
    epochs = 5
    batch_words = 10000

    temp_sen = ""
    for content in sentences:
        temp_sen = temp_sen + content[0]
    word_cut = jieba.cut(temp_sen)

    word_cut_list = remove_stop_words('remove_format.txt',word_cut)
    word_cut_list = list(set(word_cut_list))
    with open('word_cut_output.txt', 'w' ,encoding='utf-8') as file:
        for line in word_cut_list:
            file.write(line + '\n')
    train_data = word2vec.LineSentence('word_cut_output.txt')
    model = word2vec.Word2Vec(
        train_data,
        min_count=min_count,
        vector_size=vector_size,
        workers=workers,
        epochs=epochs,
        window=window_size,
        sg=sg,
        seed=seed,
        batch_words=batch_words
    )
    model.save('word2vec.model')

    padded_word_vectors = word2vec_chang(filename)

    # 將文本資料轉換成向量
    sentence_vectors = padded_word_vectors

    X = np.array(sentence_vectors).reshape(len(data), -1)

    # 將資料分成訓練集和測試集
    train_X, test_X, train_y, test_y = train_test_split(X, data['是否具有創意'].values, test_size=lstm_test_set_size,
                                                        shuffle=True, random_state=lstm_random_seed)
    train_X_reshaped = np.expand_dims(train_X, axis=-1)
    # print(train_X_reshaped.shape)
    # time.sleep(99)
    print(len(train_X), 'train examples')
    print(len(test_X), 'test examples')

    # 建模

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(activation='relu', filters=64, kernel_size=3, input_shape=
        (train_X_reshaped.shape[1], 1)),
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
    # 訓練開始時間
    start_time = time.time()
    # 假設原始輸入數據為 train_X
    # print(train_X.shape[0])
    # print(train_X.shape[1])
    # print(train_X.shape[2])
    # time.sleep(99)
    # 將輸入數據的形狀從 (None, 200, 100) 轉換為 (None, 200, 1)
    # print(train_X.shape) # 932 200 100
    # target_shape = (train_X.shape[0], -1, 1)  # 自動計算第一個維度
    # train_X_reshaped = train_X.reshape(target_shape)
    # train_X_reshaped = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)

    history = model.fit(train_X_reshaped,train_y,
              batch_size = BATCH_SIZE,
              epochs = EPOCH)
    # 訓練結束時間
    end_time = time.time()

    # 計算訓練時間（秒）
    training_time = end_time - start_time

    # 測試開始時間
    test_start_time = time.time()

    history_pd = pd.DataFrame.from_dict(history.history, orient='index')
    history_pd.columns = col
    history_pd.to_excel('word2vec_lstm_output.xlsx')

    print(history.history['accuracy']) # [0.0, 0.0, 0.0]
    print(history.history['TP']) # [0.0, 0.0, 0.0]
    print(history.history['FP']) # [0.0, 0.0, 0.0]
    print(history.history['TN']) # [0.0, 0.0, 0.0]
    print(history.history['FN']) # [0.0, 0.0, 0.0]

    #取某一層輸出新建一個model
    dense3_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_3').output)
    #以這個model預測值作為輸出
    dense3_output = dense3_layer_model.predict(train_X)
    # print(dense3_output)
    # time.sleep(99)
    fpr_3, tpr_3, thresholds_keras = roc_curve(train_y, dense3_output.ravel())
    roc_auc_3 = auc(fpr_3, tpr_3)



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

    with open('./word2vec_Report.txt','a') as f:
        f.writelines("LSTM model train report:")
        f.write("訓練時間：{} 秒 \n".format(training_time))
        f.write("測試時間：{} 秒 \n".format(testing_time))
        f.writelines("AUC: {:.4f} \n".format(roc_auc_3))
        f.writelines("Accuracy: {:.4f} \n".format(avg_acc))
        f.writelines("Precision: {:.4f} \n".format(avg_pre))
        f.writelines("Recall: {:.4f} \n".format(avg_reca))
        f.writelines("F1: {:.4f} \n".format(avg_f1))
        f.write("\n")
    model.save('lstm_word2vec.h5')
    print("LSTM模型運算完成")
    return fpr_3, tpr_3, roc_auc_3