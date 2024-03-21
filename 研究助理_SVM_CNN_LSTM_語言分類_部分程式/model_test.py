# -*- coding: utf-8 -*-
from word2vec_lstm import remove_stop_words
import time
import jieba
import numpy as np
from gensim.models import Word2Vec
import csv
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn import svm
import pandas as pd
from keras.models import Model
from joblib import load
def write_to_csv(content_list, label_list, ouput_name):
    transfer_list = []
    for temp in content_list:
        transfer_list.append(str(temp[0]))
    # 檔案名稱
    filename = 'predict_file\\'+ouput_name

    # 欄位名稱
    fieldnames = ['Content', 'Label']

    # 開啟 CSV 檔案並寫入資料
    with open(filename, 'w', newline='',encoding = 'utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 寫入表頭
        writer.writeheader()

        # 寫入資料
        for content, label in zip(transfer_list, label_list):
            writer.writerow({'Content': content, 'Label': label})

    print(f"CSV 檔案已成功輸出到 {filename}")

def adjust_lists(lists, size):
    adjusted_lists = []

    for lst in lists:
        adjusted_lst = lst[:size] + [0] * (size - len(lst)) if len(lst) < size else lst[:size]
        adjusted_arr = np.array(adjusted_lst)
        if (len(adjusted_lists) == size):
            return adjusted_lists
        adjusted_lists.append(adjusted_arr)

    while len(adjusted_lists) < size:
        adjusted_lists.append(np.zeros_like(lists[0]))

    return adjusted_lists

def readfile(file_name):
    # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
    content_list = []
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 去除表頭
        for row in csvreader:
            temp = []
            temp.append(row[0])
            content_list.append(temp)
        return content_list

def cnn_word2vec_predict(filename):

    def preprocess_text(text):
        # 預處理文本，將其轉換為符合模型要求的格式
        word2vec_model = Word2Vec.load('word2vec.model')

        word_list = jieba.lcut(text)
        word_list = remove_stop_words('remove_format.txt', word_list)
        word_vectors = [word2vec_model.wv[word] for word in word_list if word in word2vec_model.wv]
        max_length = 200
        if (len(word_vectors) == 0):
            shape = (1, 200, 100)
            empty_array = np.zeros(shape)
            return empty_array
        padded_word_vectors = adjust_lists(word_vectors, max_length)
        return np.array([padded_word_vectors])

    def model_predict(article):
        # 載入模型
        model = tf.keras.models.load_model('cnn_word2vec.h5')

        # 預處理輸入文章
        processed_article = preprocess_text(article)

        # 進行預測
        predictions = model.predict(processed_article)

        # 判斷是否具有創意
        if predictions > 0.5:
            # print("具有創意")
            return "有創意"
        else:
            # print("無創意")
            return "無創意"

    predict_label = []
    article_list = readfile(filename)
    for number in range(len(article_list)):
        temp = str(article_list[number])
        predict_label.append(model_predict(temp))
    write_to_csv(article_list,predict_label,"cnn_word2vec_predict.csv")

def lstm_word2vec_predict(filename):
    # 註冊自定義物件
    custom_objects = {'f1': tfa.metrics.F1Score}
    num_classes = 2
    def preprocess_text(text):
        # 預處理文本，將其轉換為符合模型要求的格式
        word2vec_model = Word2Vec.load('word2vec.model')

        word_list = jieba.lcut(text)
        word_list = remove_stop_words('remove_format.txt', word_list)
        word_vectors = [word2vec_model.wv[word] for word in word_list if word in word2vec_model.wv]
        max_length = 200
        if (len(word_vectors) == 0):
            shape = (1, 20000)
            empty_array = np.zeros(shape)
            X = np.array(empty_array).reshape(-1)
            return X

        padded_word_vectors = adjust_lists(word_vectors, max_length)
        padded_word_vectors = np.array(padded_word_vectors)
        X = np.array(padded_word_vectors).reshape(-1)
        return X

    def model_predict(article):
        # 載入模型
        model = tf.keras.models.load_model('lstm_word2vec.h5', custom_objects=custom_objects, compile=False)

        # 預處理輸入文章
        processed_article = preprocess_text(article)

        padded_word_vectors = np.expand_dims(processed_article, axis=0)
        # # 進行預測
        # 取某一層輸出新建一個model
        dense3_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer('Dense_3').output)
        # 以這個model預測值作為輸出
        dense3_output = dense3_layer_model.predict(padded_word_vectors)
        # 判斷是否具有創意
        if dense3_output > 0.5:
            # print("具有創意")
            return "有創意"
        else:
            # print("無創意")
            return "無創意"

    predict_label = []
    article_list = readfile(filename)
    for number in range(len(article_list)):
        temp = str(article_list[number])
        predict_label.append(model_predict(temp))
    write_to_csv(article_list,predict_label,"lstm_word2vec_predict.csv")

def svm_word2vec_predict(filename):

    def preprocess_text(text):
        # 預處理文本，將其轉換為符合模型要求的格式
        word2vec_model = Word2Vec.load('word2vec.model')

        word_list = jieba.lcut(text)
        word_list = remove_stop_words('remove_format.txt', word_list)
        word_vectors = [word2vec_model.wv[word] for word in word_list if word in word2vec_model.wv]
        max_length = 200
        # print(len(word_vectors))
        # time.sleep(1)
        if (len(word_vectors) == 0):
            shape = (1, 20000)
            empty_array = np.zeros(shape)
            X = np.array(empty_array).reshape(-1)
            return X

        padded_word_vectors = adjust_lists(word_vectors, max_length)
        padded_word_vectors = np.array(padded_word_vectors)
        X = np.array(padded_word_vectors).reshape(-1)
        return X

    def model_predict(article):

        # 載入已訓練好的 SVM 模型
        svm_model = svm.SVC()
        svm_model = load('svm_word2vec.model')

        # 預處理輸入文章
        processed_article = preprocess_text(article)

        processed_article = processed_article.reshape(1, -1)

        # 進行預測
        predictions = svm_model.predict(processed_article)
        # print(predictions)
        # 判斷是否具有創意
        if predictions > 0.5:
            # print("具有創意")
            return "有創意"
        else:
            # print("無創意")
            return "無創意"

    predict_label = []
    article_list = readfile(filename)
    for number in range(len(article_list)):
        temp = str(article_list[number])
        predict_label.append(model_predict(temp))
    write_to_csv(article_list,predict_label,"svm_word2vec_predict.csv")

def data_preocess(filename):
    # Import packages
    import csv
    import jieba
    import pandas as pd
    from collections import Counter
    import math
    import numpy as np
    import os

    # Functioins
    #-----以下函式功能：將符合格式的csv檔案，轉換為字詞矩陣
    def readfile(file_name):
        # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
        content_list = []
        with open (file_name,'r',encoding = 'utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader) # 去除表頭
            for row in csvreader:
                temp=[]
                temp.append(row[0])
                content_list.append(temp)
            return content_list

    def content_list_to_string(content_list):
        # 傳入存有每個row 資料的list，回傳所有內容結合在一起的一個string
        content_combine = ""
        for content in content_list :
            content_combine = content_combine + "".join(content)
        return content_combine

    def cut_sentence(document):
        # 傳入一個string，回傳一個經由jieba 斷詞後所產生的counter 類型(counter 中包含字詞及其出現次數)
        text = document
        words = jieba.cut(text)
        word_counts = Counter(list(words))
        return word_counts

    def convert_to_df(word_counts):
        # 將傳入的counter 類別，依照字詞出現次數多到少轉換為dataframe 的形式，並回傳此排序過的dataframe
        data = []
        for word, count in word_counts.most_common():
            data.append(pd.Series(count, name=word))
        sorted_dataframe = pd.concat(data, axis=1, sort=False)
        return sorted_dataframe

    def to_word_vector(file_name,chunk_size):
        # # Variable
        counter = 0 # counter 表示寫入csv 檔的index 值
        check_header = False # check_header 判斷是否已有表頭，False 則表尚未寫過表頭

        # 對檔案中所有內容斷詞，並統計每個詞的出現次數，以進行後續排序用
        content_list = readfile(file_name)
        # print(content_list)
        # time.sleep(10)
        total_content_word_counter = cut_sentence(content_list_to_string(content_list))
        total_content = {}
        total_content_df = pd.DataFrame(total_content)
        total_content_df = convert_to_df(total_content_word_counter)
        # 每寫chunk_size 個columns 數，就寫成一個csv 檔，總column 數除chunk_size 取整數+1 即n_chunks 數，表示會有幾個csv 檔
        n_chunks = total_content_df.shape[1] // chunk_size + 1

        # 以迴圈執行資料筆數次，row_number 代表第幾row 的資料
        for row_number in range(len(content_list)):
            if(len(content_list[row_number]) <= 0): # 判定是否有空的row，有則跳過
                continue
            counter += 1
            current_row_df = convert_to_df(cut_sentence(content_list[row_number][0]))
            current_row_df = current_row_df.reindex(columns = total_content_df.columns, fill_value = 0) # 調整當前row_df 的column 數到跟所有內容的dataframe 數量一致，若沒有出現的字詞則填"0"
            current_row_df.index += counter
            for i in range(n_chunks): # 對整個dataframe 資料，分成n 個區塊寫入csv 檔
                if(check_header == False): # check_header = False 時，則寫入表頭
                    #用.iloc[row,column]取出要寫入csv 的資料，取出 i*chunk_size 到第(i+1)*chunk_size 個column 寫入csv
                    current_row_df.iloc[:, i*chunk_size:(i+1)*chunk_size].to_csv(temp_filename+'\\'+f"tokenize_{i}.csv", encoding = 'utf-8-sig', header = True, mode = 'a')
                else:
                    current_row_df.iloc[:, i*chunk_size:(i+1)*chunk_size].to_csv(temp_filename+'\\'+f"tokenize_{i}.csv", encoding = 'utf-8-sig', header = False, mode = 'a')
            check_header = True
            total_content_df = current_row_df # 給下一個row 的dataframe 參照其column 數

    #-----以下函式功能：利用tokenize.csv檔，計算各字詞出現在所有content的次數，並生成tf-idf運算所需之i值
    def read_word(file_name):
        # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
        word_list = []
        with open (temp_filename+'\\'+file_name,'r',encoding = 'utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                for word in range(len(row)-1):
                    word_list.append(row[word+1])
                break
            return word_list

    def count_idf_origin(file_name,word_list):
        # 計算i值，產生i值之csv檔
        print("計算i值...")
        idf_dict = {}
        for number_word in range(len(word_list)): # 此迴圈用於遍歷每個字元(即csv檔案中第一個row由左至右的每一個欄位:數量上限為10000)
            with open (temp_filename+'\\'+file_name,'r',encoding = 'utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader) # 去除表頭
                idf_count = 0
                for row in csvreader: # 此迴圈用於遍歷當前字元在所有文章的出現次數
                    if(int(row[number_word])>0): #此判斷式用於判斷此目標字元，在當前的文章是否出現過
                        idf_count+=1 #有則+1
            idf_dict[word_list[number_word]] = idf_count
            if(number_word%1000 == 1 ):
                print(idf_dict)
                print("進度: +1000")
        with open(temp_filename+'\\'+'idf_i_value.csv', 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            # 寫入表頭
            writer.writerow(['Key', 'Value'])
            # 遍歷每一組字典
            for d in idf_dict:
                # 遍歷該字典的鍵和值
                writer.writerow([d, idf_dict.get(d)])

    def count_idf(file_name, word_list):
        # 計算i值，產生i值之csv檔
        print("計算i值...")
        idf_dict = {word: 0 for word in word_list}
        with open(temp_filename+'\\'+file_name, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # 去除表頭
            for row in csvreader:
                for i, count in enumerate(row):
                    if i == 0:
                        continue
                    if int(count) > 0:
                        idf_dict[word_list[i-1]] += 1
        with open(temp_filename+'\\'+'idf_i_value.csv', 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            # 寫入表頭
            writer.writerow(['Key', 'Value'])
            # 遍歷每一組字典
            for d in idf_dict:
                # 遍歷該字典的鍵和值
                writer.writerow([d, idf_dict.get(d)])

    #-----以下函式功能：利用idf_i_value計算後得到的i值，加以計算出idf值
    def read_i_list(file_name):
        # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
        df = pd.read_csv(temp_filename+'\\'+file_name)
        i_list = []
        i_list = df['Value'].tolist()
        for index in range(len(i_list)):
            if(i_list[index] == 0):
                i_list[index] = 1
        return i_list

    def caculat_idf(idf_i_list,d):
        with open(temp_filename+'\\'+'idf_score.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            for i in idf_i_list:
                temp = 0
                temp = math.log(d/i) #以10為底，以(D/i)取log
                writer.writerow((temp,))

    #-----以下函式功能：將每個文本的每個字進行tf-idf運算，生成tf-idf矩陣
    def read_data(file_1,file_2):
        # file_1 為：content_total_word_number.csv , file_2 為：idf_score.csv
        with open(temp_filename+'\\'+file_1, 'r', encoding='utf-8-sig') as readfile_1:
            total_word_number = readfile_1.readlines()
        with open(temp_filename+'\\'+file_2, 'r', encoding='utf-8-sig') as readfile_2:
            idf_score = readfile_2.readlines()
        return total_word_number,idf_score

    def tf_idf_convert(filename, total_word_number, idf_score):
        data = pd.read_csv(temp_filename+'\\'+filename)
        cloumns_list = list(data.columns)
        cloumns_list.remove(cloumns_list[0])
        data = data.drop(data.columns[0], axis=1)
        data = np.array(data)
        total_word_number = np.array(total_word_number)
        idf_score = np.array(idf_score)
        tf_idf = np.zeros_like(data, dtype=float)
        for i in range(data.shape[0]):
            tf_idf[i] = data[i].astype(float) / total_word_number[i].astype(float) * idf_score
        data = pd.DataFrame(tf_idf)
        data.columns = cloumns_list
        data.to_csv(temp_filename+'\\'+'tf_idf_matrix.csv', encoding = 'utf-8-sig', index='true')

    def content_total_word(content_list):
        # 計算斷詞總共分為幾個字詞
        for text in content_list:
            total_number = 0
            words = jieba.cut(text[0])
            total_number = str(len(list(words)))
            with open(temp_filename+'\\'+'content_total_word_number.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow((total_number,))

    def concat_csv(file_1,file_2):
        # 將 tf-idf矩陣以及label文檔進行concat
        csv1_df = pd.read_csv(temp_filename+'\\'+file_1, header=None ,low_memory=False)
        csv2_df = pd.read_csv(file_2, usecols=[1],header=None, low_memory=False)
        result_df = pd.concat([csv1_df, csv2_df], axis=1)
        result_df.to_csv(temp_filename+'\\'+'result.csv', encoding = 'utf-8-sig', index=False, header=None) #寫成csv，file_1需輸入tf_idf_matrix；file_2需輸入具有label的原始文件
        return result_df

    # # Setting
    file_name = filename
    chunk_size = 9999999  # chunk_size 表示每寫入幾個字詞，就換到下一個csv 檔
    total_content_number = len(readfile(filename))  # 總樣本數

    # # Variable
    total_word_number = []
    idf_score = []
    content_list = []

    temp_filename = "tfidf_predict_temp_data"
    # # Main function
    if not os.path.exists(temp_filename):
        os.mkdir(temp_filename)      # 創建新資料夾儲存檔案
    else:
        print(temp_filename+" 資料夾已存在，若此資料夾內容不足6個檔案，則請刪除此資料夾並重新執行")
        return

    to_word_vector(file_name, chunk_size)  # 將原始資料轉換成詞頻矩陣
    print("tokenize_0.csv 成功輸出")

    word_list = read_word("tokenize_0.csv")  # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
    count_idf("tokenize_0.csv", word_list)  # 計算i值，產生i值之csv檔
    print("idf_i_value.csv 成功輸出")

    idf_i_list = read_i_list("idf_i_value.csv")  # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
    caculat_idf(idf_i_list, total_content_number)  # 計算每idf之值，並將其計算結果存為csv
    print("idf_score.csv 成功輸出")

    content_list = readfile(file_name)
    content_total_word(content_list)  # 計算斷詞總共分為幾個字詞
    print("content_total_word_number.csv 成功輸出")

    list_tuple = read_data('content_total_word_number.csv', 'idf_score.csv')
    total_word_number = list(map(float, list_tuple[0]))
    idf_score = list(map(float, list_tuple[1]))
    tf_idf_convert('tokenize_0.csv', total_word_number, idf_score)  # 將矩陣進行tf-idf轉換
    print("tf_idf_matrix.csv 成功輸出")

    concat_csv('tf_idf_matrix.csv', file_name)  # 將 tf-idf矩陣以及label文檔進行concat
    print("result.csv 成功輸出")

def cnn_tfidf_predict(filename):
    data_preocess(filename)
    import numpy as np

    def adjust_array_shape(arr, target_shape):
        current_shape = arr.shape
        pad_width = [(0, 0) for _ in range(len(target_shape))]  # 設定填充寬度的初始值為 (0, 0) 的元組列表

        # 計算每個維度的填充或刪除寬度
        for i in range(len(target_shape)):
            if current_shape[i] < target_shape[i]:
                pad_width[i] = (0, target_shape[i] - current_shape[i])
            elif current_shape[i] > target_shape[i]:
                arr = np.delete(arr, np.s_[target_shape[i]:], axis=i)

        # 填充或刪除操作
        arr = np.pad(arr, pad_width, mode='constant', constant_values=0)

        return arr

    def model_predict():
        # 載入模型
        model = tf.keras.models.load_model('cnn_tfidf.h5')

        concat_df = pd.read_csv('tfidf_predict_temp_data'+'\\'+'result.csv', header=None, low_memory=False)  # 讀取已進行tf-idf轉換，且將標籤結合的矩陣的檔案

        concat_df.iloc[0, 0] = '字詞特徵'
        new_col = concat_df.iloc[0, :]

        concat_df.columns = new_col
        concat_df = concat_df.iloc[1:, 1:]  # 去除表頭和列名

        X_train = concat_df.astype(np.float32)
        tensor_X = tf.convert_to_tensor(X_train)
        tfidf_X = np.expand_dims(tensor_X, axis=-1)  # 將 TF-IDF 特徵轉換為 3D 張量，為了要符合conv的input需求，須將2D轉3D
        X_test_tfidf = np.expand_dims(tfidf_X, axis=-1)

        print(X_test_tfidf.shape[1])
        target_shape = (X_test_tfidf.shape[0], shape_number, 1,1)

        adj_arr = adjust_array_shape(X_test_tfidf,target_shape)
        # 進行預測
        predictions = model.predict(adj_arr)
        return predictions

    predict_label = []
    predictions = model_predict()
    article_list = readfile(filename)
    for number in range(len(predictions)):
        # print(predictions[number][0])
        if (predictions[number][0] > 0.5):
            predict_label.append('有創意')
        else:
            predict_label.append('無創意')
    write_to_csv(article_list,predict_label,"cnn_tfidf_predict.csv")

def lstm_tfidf_predict(filename):
    data_preocess(filename)
    import numpy as np
    custom_objects = {'f1': tfa.metrics.F1Score}
    num_classes = 2
    def adjust_array_shape(arr, target_shape):
        current_shape = arr.shape
        pad_width = [(0, 0) for _ in range(len(target_shape))]  # 設定填充寬度的初始值為 (0, 0) 的元組列表

        # 計算每個維度的填充或刪除寬度
        for i in range(len(target_shape)):
            if current_shape[i] < target_shape[i]:
                pad_width[i] = (0, target_shape[i] - current_shape[i])
            elif current_shape[i] > target_shape[i]:
                arr = np.delete(arr, np.s_[target_shape[i]:], axis=i)

        # 填充或刪除操作
        arr = np.pad(arr, pad_width, mode='constant', constant_values=0)

        return arr

    def model_predict():
        # 載入模型
        model = tf.keras.models.load_model('lstm_tfidf.h5', custom_objects=custom_objects, compile=False)

        concat_df = pd.read_csv('tfidf_predict_temp_data'+'\\'+'result.csv', header=None, low_memory=False)  # 讀取已進行tf-idf轉換，且將標籤結合的矩陣的檔案

        concat_df.iloc[0, 0] = '字詞特徵'
        new_col = concat_df.iloc[0, :]

        concat_df.columns = new_col
        concat_df = concat_df.iloc[1:, 1:]  # 去除表頭和列名


        X_train = concat_df.astype(np.float32)
        tensor_X = tf.convert_to_tensor(X_train)
        tfidf_X = np.expand_dims(tensor_X, axis=-1)  # 將 TF-IDF 特徵轉換為 3D 張量，為了要符合conv的input需求，須將2D轉3D
        X_test_tfidf = np.expand_dims(tfidf_X, axis=-1)

        target_shape = (X_test_tfidf.shape[0],shape_number,1,1)
        adj_arr = adjust_array_shape(X_test_tfidf,target_shape)
        # 進行預測
        predictions = model.predict(adj_arr)
        return predictions

    predict_label = []
    predictions = model_predict()
    article_list = readfile(filename)
    for number in range(len(predictions)):
        # print(predictions[number][0])
        if (predictions[number][0] > 0.47):
            predict_label.append('有創意')
        else:
            predict_label.append('無創意')
    write_to_csv(article_list,predict_label,"lstm_tfidf_predict.csv")

def svm_tfidf_predict(filename):
    data_preocess(filename)
    import numpy as np

    def adjust_array_shape(arr, target_shape):
        current_shape = arr.shape
        pad_width = [(0, 0) for _ in range(len(target_shape))]  # 設定填充寬度的初始值為 (0, 0) 的元組列表

        # 計算每個維度的填充或刪除寬度
        for i in range(len(target_shape)):
            if current_shape[i] < target_shape[i]:
                pad_width[i] = (0, target_shape[i] - current_shape[i])
            elif current_shape[i] > target_shape[i]:
                arr = np.delete(arr, np.s_[target_shape[i]:], axis=i)

        # 填充或刪除操作
        arr = np.pad(arr, pad_width, mode='constant', constant_values=0)

        return arr

    def model_predict():
        # 載入模型
        # 載入已訓練好的 SVM 模型
        svm_model = svm.SVC()
        svm_model = load('svm_tfidf.model')

        concat_df = pd.read_csv('tfidf_predict_temp_data'+'\\'+'result.csv', header=None, low_memory=False)  # 讀取已進行tf-idf轉換，且將標籤結合的矩陣的檔案

        concat_df.iloc[0, 0] = '字詞特徵'
        new_col = concat_df.iloc[0, :]

        concat_df.columns = new_col
        concat_df = concat_df.iloc[1:, 1:]  # 去除表頭和列名

        target_shape = (concat_df.shape[0],shape_number)
        adj_arr = adjust_array_shape(concat_df,target_shape)
        # 進行預測
        predictions = svm_model.predict(adj_arr)
        return predictions

    predict_label = []
    predictions = model_predict()
    article_list = readfile(filename)

    for number in range(len(predictions)):
        if (float(predictions[number]) > 0.5):
            predict_label.append('無創意')
        else:
            predict_label.append('有創意')
    write_to_csv(article_list,predict_label,"svm_tfidf_predict.csv")

def bert_predict(filename):
    from transformers import BertTokenizer
    import torch
    from torch import nn
    from transformers import BertModel
    class BertClassifier(nn.Module):
        def __init__(self, dropout=0.5):
            super(BertClassifier, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(768, 2)
            # self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax()

        def forward(self, input_id, mask):
            _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
            dropout_output = self.dropout(pooled_output)
            linear_output = self.linear(dropout_output)
            # final_layer = self.sigmoid(linear_output)
            final_layer = self.softmax(linear_output)
            return final_layer

    def predict(model,tokenizer,input_file, output_file):
        # 判斷是否使用 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # 讀取輸入檔案
        df = pd.read_csv(input_file)

        # 初始化預測結果的列表
        predictions = []

        # 初始化 BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # 迭代每一行文章
        for i, row in df.iterrows():
            # 取得文章內容
            text = row['Content']

            # 將文章轉換成 BERT 的輸入格式
            inputs = tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True
            )

            # 將輸入資料轉換成 Tensor 格式
            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device)

            # 進行預測
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                predictions.append(outputs.argmax().item())

        # 將預測結果轉換成 DataFrame
        predictions_df = pd.DataFrame({'prediction': predictions})
        # 將預測結果存成新的 CSV 檔案
        predictions_df.to_csv(output_file, index=False)

    model = BertClassifier()
    # 載入訓練好的權重
    model.load_state_dict(torch.load('bert_model.pt'))

    input_file = filename  # 輸入的 CSV 檔案
    output_file = 'bert_predictions.csv'  # 輸出的 CSV 檔案

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    predict(model, tokenizer, input_file, output_file)

    # 讀取兩個 CSV 檔案
    csv1 = pd.read_csv('bert_predictions.csv')
    csv2 = pd.read_csv(filename)

    # 取得 file2.csv 的第一個欄位內容
    content_column = csv2.iloc[:, 0]

    # 將 file1.csv 的第一個欄位改為第二個欄位
    label_column = csv1.iloc[:, 0]

    # 將兩個欄位進行連接
    concatenated_column = pd.concat([content_column, label_column], axis=1)
    concatenated_column['prediction'] = concatenated_column['prediction'].replace(0, '無創意')
    concatenated_column['prediction'] = concatenated_column['prediction'].replace(1, '有創意')
    # 儲存連接後的結果
    concatenated_column.to_csv('predict_file\\'+'bert_predict.csv', index=False,encoding = 'utf-8-sig')

def main():
    import os
    filename = input("請輸入文本的完整檔名(包含副檔名) ex:環保創意模型分類(380).csv :")
    # filename='環保創意模型分類(200_刪除空格).csv'
    if not os.path.exists('predict_file'):
        os.mkdir('predict_file')      # 創建新資料夾儲存檔案
    global shape_number
    concat_df = pd.read_csv('temp_data\\' + 'result.csv', header=None, low_memory=False) # 讀取已進行tf-idf轉換，且將標籤結合的矩陣的檔案

    concat_df.iloc[0, 0] = '字詞特徵'
    new_col = concat_df.iloc[0, :]
    shape_number = len(new_col) - 2


    cnn_word2vec_predict(filename)
    svm_word2vec_predict(filename)
    lstm_word2vec_predict(filename)

    cnn_tfidf_predict(filename)
    lstm_tfidf_predict(filename)
    svm_tfidf_predict(filename)

    # bert_predict(filename)
    print("程式執行完畢")

if __name__ == '__main__':
    main()
