# -*- coding: utf-8 -*-
from tensorflow.keras import layers
from gensim.models import word2vec
import time
import os
from datetime import datetime
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from word2vec_lstm import train_lstm,remove_stop_words
from sklearn.svm import SVC
from sklearn import svm
import jieba
import pickle
def train_svm():

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

    sentence_vectors = np.array(sentence_vectors)
    # print(sentence_vectors.shape)
    X = np.array(sentence_vectors).reshape(len(data), -1)
    # print(X.shape)
    # time.sleep(99)
    # 將資料分成訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, data['是否具有創意'].values, test_size=svm_test_set_size, random_state = svm_random_seed)
    # 建立SVM分類器並進行訓練
    svc_model = svm.SVC()
    if(kernal_mode == 'LinearSVC'):
        svc_modle = svm.LinearSVC(C = c_value, max_iter = 10000)
    elif(kernal_mode == 'linear'):
        svc_modle = SVC(kernel = kernal_mode, C = c_value, random_state = svm_random_seed)
    elif(kernal_mode == 'poly'):
        svc_modle = SVC(kernel = kernal_mode, degree = degree_value, gamma = gamma_value, C = c_value, random_state = svm_random_seed)
    else:
        svc_modle = SVC(kernel = kernal_mode, gamma = gamma_value, C = c_value, random_state = svm_random_seed)


    start_time = time.time()  # 開始訓練時間
    svc_model.fit(X_train, y_train)
    end_time = time.time() # 訓練完成時間
    train_time = end_time - start_time # 計算訓練時間

    start_time = time.time()  # 開始測試時間
    scores = svc_model.score(X_test, y_test)  # 開始測試
    end_time = time.time() # 測試完成時間
    test_time = end_time - start_time # 計算測試時間

    print("訓練時間：{} 秒".format(train_time))
    print("測試時間：{} 秒".format(test_time))
    with open('svm_word2vec.model', 'wb') as f:
        pickle.dump(svc_model, f)
    # 進行預測
    y_pred = svc_model.predict(X_test)

    # 計算各項指標的值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # 輸出結果
    print("AUC: {:.4f}".format(roc_auc))
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))

    with open('./word2vec_Report.txt','a') as f:
        f.writelines("SVM model train report:")
        f.write("訓練時間：{} 秒 \n".format(train_time))
        f.write("測試時間：{} 秒 \n".format(test_time))
        f.writelines("AUC: {:.4f} \n".format(roc_auc))
        f.writelines("Accuracy: {:.4f} \n".format(acc))
        f.writelines("Precision: {:.4f} \n".format(precision))
        f.writelines("Recall: {:.4f} \n".format(recall))
        f.writelines("F1: {:.4f} \n".format(f1))
        f.write("\n")

    print("SVM模型運算完成")
    return fpr, tpr, roc_auc

def train_cnn():
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

    sentence_vectors = np.array(sentence_vectors)

    X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, data['是否具有創意'].values,
                                                        test_size=test_set_size, random_state=random_seed)
   # 建立CNN模型
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(200, 100)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 只有兩個分類，使用sigmoid作為輸出層的激活函數

    # 編譯模型
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time() # 開始訓練時間
    # # 模型訓練
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    end_time = time.time() # 訓練完成時間
    model.save('cnn_word2vec.h5')
    train_time = end_time - start_time # 計算訓練時間

    # 評估模型
    start_time = time.time()  # 開始測試時間
    y_pred = model.predict(X_test)
    # print(y_pred[0:10])
    threshold = 0.5
    predictions = model.predict(X_test)  # 假設 input_data 是輸入資料
    predicted_labels = (predictions > threshold).astype(int)
    # print(predicted_labels)
    # time.sleep(5)
    end_time = time.time()  # 測試完成時間
    test_time = end_time - start_time  # 計算測試時間
    # 將預測的概率轉換為二元類別
    y_pred_binary = np.round(y_pred)

    # 計算accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)

    # 計算precision
    precision = precision_score(y_test, y_pred_binary)

    # 計算recall
    recall = recall_score(y_test, y_pred_binary)

    # 計算F1 Score
    f1 = f1_score(y_test, y_pred_binary)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # 輸出結果
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))
    print("AUC: {:.4f}".format(roc_auc))

    with open('./word2vec_Report.txt','a') as f:
        f.writelines("CNN model train report:")
        f.write("訓練時間：{} 秒 \n".format(train_time))
        f.write("測試時間：{} 秒 \n".format(test_time))
        f.writelines("AUC: {:.4f} \n".format(roc_auc))
        f.writelines("Accuracy: {:.4f} \n".format(accuracy))
        f.writelines("Precision: {:.4f} \n".format(precision))
        f.writelines("Recall: {:.4f} \n".format(recall))
        f.writelines("F1: {:.4f} \n".format(f1))
        f.write("\n")

    print("CNN模型運算完成")

    return fpr, tpr, roc_auc

# ROC 曲線
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    current_time = datetime.now().strftime("%m%d~%H%M")
    plt.savefig('curve_picture\\' + 'roc_curve' + '_' + str(lr_value) + '_' + str(random_seed) + '_' + str(current_time) + '.png')
    plt.show()

def plot_merge_curve(fpr, tpr, roc_auc,fpr_2, tpr_2, roc_auc_2,fpr_3, tpr_3, roc_auc_3):
    # 繪製第一張 ROC 圖
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='SVM_ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Word2vec_Receiver operating characteristic')
    plt.legend(loc="lower right")

    # 繪製第二張 ROC 圖
    plt.plot(fpr_2, tpr_2, color='green', lw=2, label='CNN_ROC curve (area = %0.2f)' % roc_auc_2)
    plt.xlim([0.0, 1.0]) # 設定 x 軸範圍相同
    plt.ylim([0.0, 1.05]) # 設定 y 軸範圍相同
    plt.legend(loc="lower right")

    # 繪製第三張 ROC 圖
    plt.plot(fpr_3, tpr_3, color='red', lw=2, label='LSTM_ROC curve (area = %0.2f)' % roc_auc_3)
    plt.xlim([0.0, 1.0]) # 設定 x 軸範圍相同
    plt.ylim([0.0, 1.05]) # 設定 y 軸範圍相同
    plt.legend(loc="lower right")
    current_time = datetime.now().strftime("%m%d~%H%M")

    plt.savefig('word2vec_curve_picture\\' + 'word2vec_merge' +str(current_time)+ '.png')

    # 顯示三張圖片疊加後的結果
    plt.show()
def main():
    # # SVM_Variable
    global c_value
    global svm_random_seed
    global kernal_mode
    global svm_labels_name
    global svm_test_set_size
    global gamma_value
    global degree_value

    # # SVM_Setting
    c_value = 10 # C值越大模型學習效果越好，越小則學習效果降低，預設為1
    svm_random_seed = 101 # 固定隨機種子數值，使輸出可以固定
    kernal_mode = 'rbf' # Svm 可以設定的模式有：LinearSVC、linear、poly、rbf
    gamma_value = 0.5 # 利用rbf、poly模式時可設定，可設置為'scale'，或是其他小數如:0.7、0.3...(數值越大越能做複雜的分類邊界)
    degree_value = 2 # 當模式為poly時，可以設置degree程度，可增加模型複雜度，3 代表轉換到三次空間進行分類
    svm_test_set_size = 0.3 # 測試集占比大小
    svm_labels_name = '是否具有創意' # 原始文本中，標籤的表頭名稱

    # # CNN_Variable
    global lr_value
    global random_seed
    global labels_name
    global test_set_size
    global BATCH_SIZE
    global EPOCHS
    # # CNN_Setting
    lr_value = 0.0001  # 學習率越高，模型學習效果越好，越小則學習效果降低
    random_seed = 101  # 固定隨機種子數值，使輸出可以固定
    test_set_size = 0.3  # 測試集占比大小
    EPOCHS = 4
    BATCH_SIZE = 8
    labels_name = '是否具有創意'  # 原始文本中，標籤的表頭名稱

    # # Main function
    if not os.path.exists('word2vec_curve_picture'):
        os.mkdir('word2vec_curve_picture')
    # 測試
    # fpr_2, tpr_2, roc_auc_2 = train_lstm(filename)
    # plot_roc_curve(fpr_2, tpr_2, roc_auc_2)

    fpr_2, tpr_2, roc_auc_2 = train_cnn()
    fpr, tpr, roc_auc = train_svm()
    fpr_3, tpr_3, roc_auc_3 = train_lstm(filename)
    plot_merge_curve(fpr, tpr, roc_auc, fpr_2, tpr_2, roc_auc_2,fpr_3, tpr_3, roc_auc_3)

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

if __name__ == '__main__':
    # test()
    global padded_word_vectors
    global filename
    filename = input("請輸入文本的完整檔名(包含副檔名) ex:環保創意評價(1332) (final).csv :")
    # filename = '環保創意評價(1332) (final).csv'
    main()