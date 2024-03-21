import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.svm import SVC
from sklearn import svm
from tfidf_lstm import train_lstm
import pickle
def train_SVM(filename):
    print('模型運算中...')
    cncat_df = pd.read_csv('temp_data\\'+filename, header=None ,low_memory=False)
    
    cncat_df.iloc[0,0] = '字詞特徵'
    new_col = cncat_df.iloc[0, :]
    print('總字詞特徵數量: '+str(len(new_col)))

    cncat_df.columns = new_col
    cncat_df = cncat_df.iloc[1:, 1:]

    X = cncat_df.drop(labels=[svm_labels_name],axis=1).values # 移除'是否具有創意'column，並取得剩下欄位資料
    y = cncat_df[svm_labels_name].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = svm_test_set_size, random_state = svm_random_seed, stratify=y)
   
    if(kernal_mode == 'LinearSVC'):
        svc_modle = svm.LinearSVC(C = c_value, max_iter = 10000)
    elif(kernal_mode == 'linear'):
        svc_modle = SVC(kernel = kernal_mode, C = c_value, random_state = svm_random_seed)
    elif(kernal_mode == 'poly'):
        svc_modle = SVC(kernel = kernal_mode, degree = degree_value, gamma = gamma_value, C = c_value, random_state = svm_random_seed)
    else:
        svc_modle = SVC(kernel = kernal_mode, gamma = gamma_value, C = c_value, random_state = svm_random_seed)

    start_time = time.time()  # 開始訓練時間
    svc_modle.fit(X_train, y_train)
    end_time = time.time() # 訓練完成時間
    train_time = end_time - start_time # 計算訓練時間

    start_time = time.time()  # 開始測試時間
    scores = svc_modle.score(X_test, y_test)  # 開始測試
    end_time = time.time() # 測試完成時間
    test_time = end_time - start_time # 計算測試時間

    y_pred = svc_modle.predict(X_test)
    # 預測機率
    y_score = svc_modle.decision_function(X_test)

    # 計算假正率和真正率以及阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label='1')

    # 計算AUC
    roc_auc = auc(fpr, tpr)

    # 計算Accuracy、Precision、Recall、F1等結果
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='1')
    recall = recall_score(y_test, y_pred,pos_label='1')
    f1 = f1_score(y_test, y_pred,pos_label='1')

    print("訓練時間：{} 秒".format(train_time))
    print("測試時間：{} 秒".format(test_time))
    with open('svm_tfidf.model', 'wb') as f:
        pickle.dump(svc_modle, f)
    # 輸出結果
    print("AUC: {:.4f}".format(roc_auc))
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))

    with open('./tfidf_Report.txt','a') as f:
        f.write("SVM model train report:")
        f.write("訓練時間：{} 秒\n".format(train_time))
        f.writelines("測試時間：{} 秒\n".format(test_time))
        f.writelines("AUC: {:.4f}\n".format(roc_auc))
        f.writelines("Accuracy: {:.4f}\n".format(acc))
        f.writelines("Precision: {:.4f}\n".format(precision))
        f.writelines("Recall: {:.4f}\n".format(recall))
        f.writelines("F1: {:.4f}\n".format(f1))
        f.write("\n")

    print("SVM模型運算完成")
    return fpr, tpr, roc_auc


def train_CNN(filename):
    print('模型運算中...')
    concat_df = pd.read_csv('temp_data\\' + filename, header=None, low_memory=False) # 讀取已進行tf-idf轉換，且將標籤結合的矩陣的檔案

    concat_df.iloc[0, 0] = '字詞特徵'
    new_col = concat_df.iloc[0, :]
    print('總字詞特徵數量: ' + str(len(new_col))) # 計算出所有字詞特徵的數量

    concat_df.columns = new_col
    concat_df = concat_df.iloc[1:, 1:] # 去除表頭和列名

    X = concat_df.drop(labels=[cnn_labels_name], axis=1).values  # 移除'是否具有創意'column，並取得剩下欄位資料
    y = concat_df[cnn_labels_name].values #　取得所有列的標籤值
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cnn_test_set_size, random_state=cnn_random_seed) # 將資料依照test_set_size 比例切分(此參數代表測試集占所有資料比)

    # 將資料numpy array 轉換成float 型態，得以和tensor 型態相容
    X_train = X_test.astype(np.float32)
    y_train = y_test.astype(np.float32)
    tensor_X = tf.convert_to_tensor(X_train)
    tensor_y = tf.convert_to_tensor(y_train)

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    tensor_X_test = tf.convert_to_tensor(X_test)
    tensor_y_test = tf.convert_to_tensor(y_test)

    # 建立 CNN 模型
    model = Sequential() # 建立一個空的 Sequential 模型
    model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=(tensor_X.shape[1], 1))) #這個層包含 128 個卷積核，每個卷積核的大小是 5，
    # 使用 relu 激活函數，並且需要輸入形狀為 (tensor_X.shape[1], 1) 的張量
    model.add(MaxPooling1D(pool_size=2)) # 將每個區域中最大的值留下，其餘的捨棄。池化大小是 2，表示每 2 個值中取一個最大值
    model.add(Flatten()) # 將前面的卷積和池化層輸出的多維張量攤平成一維向量，才能輸入到全連接層
    model.add(Dense(2, activation='softmax')) # 這個層的作用是將攤平後的一維向量映射到 2 個類別的概率分佈上。在訓練過程中，這個層的輸出會被轉換成預測標籤

    # 模型編譯
    model.compile(optimizer=Adam(lr=lr_value), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # optimizer 是 Adam 優化器，lr_value 設定學習率；loss 損失函數，這裡用稀疏分類交叉熵（sparse categorical crossentropy）；metrics 是評估指標，這裡使用accuracy

    tensor_X = np.expand_dims(tensor_X, axis=-1)  # 將 TF-IDF 特徵轉換為 3D 張量，為了要符合conv的input需求，須將2D轉3D
    X_test_tfidf = np.expand_dims(tensor_X_test, axis=-1) # 第三個維度的值皆設為1


    start_time = time.time() # 開始訓練時間
    model.fit(tensor_X, y_train, batch_size=8, epochs=4, validation_split=0.1) # 開始訓練
    end_time = time.time() # 訓練完成時間
    train_time = end_time - start_time # 計算訓練時間
    model.save('cnn_tfidf.h5')

    X_test_tfidf = np.expand_dims(X_test_tfidf, axis=-1)

    start_time = time.time()  # 開始測試時間
    # predictions = model.predict(X_test_tfidf)# 開始測試
    end_time = time.time() # 測試完成時間
    test_time = end_time - start_time # 計算測試時間

    # 輸出結果
    print("訓練時間：{} 秒".format(train_time))
    print("測試時間：{} 秒".format(test_time))

    preds = model.predict(X_test_tfidf)
    # print(y_test[0:10])
    # print(preds[0:10])
    threshold = 0.5
    predictions = model.predict(X_test_tfidf)  # 假設 input_data 是輸入資料
    predicted_labels = (predictions > threshold).astype(int)
    # print(predicted_labels)
    # time.sleep(5)
    y_pred = np.argmax(preds, axis=1)

    # 計算 ROC 曲線和 AUC
    fpr, tpr, thresholds = roc_curve(y_test, preds[:, 1])
    roc_auc = roc_auc_score(y_test, preds[:, 1])

    # 計算 Accuracy、Precision、Recall、F1 等指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 輸出結果
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))
    print("AUC: {:.4f}".format(roc_auc))

    # plot_roc_curve(fpr, tpr, roc_auc) # 繪製 ROC 曲線
    with open('./tfidf_Report.txt','a') as f:
        f.writelines("CNN model train report:")
        f.write("訓練時間：{} 秒\n".format(train_time))
        f.write("測試時間：{} 秒\n".format(test_time))
        f.writelines("AUC: {:.4f}\n".format(roc_auc))
        f.writelines("Accuracy: {:.4f}\n".format(accuracy))
        f.writelines("Precision: {:.4f}\n".format(precision))
        f.writelines("Recall: {:.4f}\n".format(recall))
        f.writelines("F1: {:.4f}\n".format(f1))
        f.write("\n")

    print("CNN模型運算完成")

    return fpr, tpr, roc_auc

def plot_merge_curve(fpr, tpr, roc_auc,fpr_2, tpr_2, roc_auc_2,fpr_3, tpr_3, roc_auc_3):
    # 繪製第一張 ROC 圖
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='SVM_ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('TF_IDF_Receiver operating characteristic')
    plt.legend(loc="lower right")
    current_time = datetime.now().strftime("%m%d~%H%M")
    # plt.savefig('curve_picture\\' + 'roc_curve' + '_' + str(c_value) + '_' + str(svm_random_seed) + '_' + str(current_time) + '.png')

    # 繪製第二張 ROC 圖
    plt.plot(fpr_2, tpr_2, color='green', lw=2, label='CNN_ROC curve (area = %0.2f)' % roc_auc_2)
    plt.xlim([0.0, 1.0]) # 設定 x 軸範圍相同
    plt.ylim([0.0, 1.05]) # 設定 y 軸範圍相同
    plt.legend(loc="lower right")
    current_time = datetime.now().strftime("%m%d~%H%M")

    # 繪製第三張 ROC 圖
    plt.plot(fpr_3, tpr_3, color='red', lw=2, label='LSTM_ROC curve (area = %0.2f)' % roc_auc_3)
    plt.xlim([0.0, 1.0]) # 設定 x 軸範圍相同
    plt.ylim([0.0, 1.05]) # 設定 y 軸範圍相同
    plt.legend(loc="lower right")
    current_time = datetime.now().strftime("%m%d~%H%M")

    plt.savefig('tfidf_curve_picture\\' + 'tfidf_merge' +str(current_time)+ '.png')

    # 顯示三張圖片疊加後的結果
    plt.show()

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
    plt.savefig('curve_picture\\' + 'roc_curve' + '_' + str(lr_value) + '_' + str(current_time) + '.png')
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
    global cnn_random_seed
    global cnn_labels_name
    global cnn_test_set_size

    # # CNN_Setting
    lr_value = 0.001  # 學習率越高，模型學習效果越好，越小則學習效果降低
    cnn_random_seed = 101  # 固定隨機種子數值，使輸出可以固定
    cnn_test_set_size = 0.3  # 測試集占比大小
    cnn_labels_name = '是否具有創意'  # 原始文本中，標籤的表頭名稱

    # # Main function
    if not os.path.exists('tfidf_curve_picture'):
        os.mkdir('tfidf_curve_picture')
    # fpr_2, tpr_2, roc_auc_2 = train_CNN('result.csv')
    fpr, tpr, roc_auc = train_SVM('result.csv')
    # fpr_3, tpr_3, roc_auc_3 = train_lstm()
    # plot_roc_curve(fpr_3, tpr_3, roc_auc_3)
    # plot_merge_curve(fpr, tpr, roc_auc,fpr_2, tpr_2, roc_auc_2,fpr_3, tpr_3, roc_auc_3)

if __name__ == '__main__':
    main()