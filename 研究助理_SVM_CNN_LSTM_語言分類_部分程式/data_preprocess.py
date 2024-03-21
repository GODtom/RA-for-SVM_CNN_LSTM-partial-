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
                current_row_df.iloc[:, i*chunk_size:(i+1)*chunk_size].to_csv('temp_data\\'+f"tokenize_{i}.csv", encoding = 'utf-8-sig', header = True, mode = 'a')
            else:
                current_row_df.iloc[:, i*chunk_size:(i+1)*chunk_size].to_csv('temp_data\\'+f"tokenize_{i}.csv", encoding = 'utf-8-sig', header = False, mode = 'a')
        check_header = True
        total_content_df = current_row_df # 給下一個row 的dataframe 參照其column 數

#-----以下函式功能：利用tokenize.csv檔，計算各字詞出現在所有content的次數，並生成tf-idf運算所需之i值
def read_word(file_name):
    # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
    word_list = []
    with open ('temp_data\\'+file_name,'r',encoding = 'utf-8') as csvfile:      
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
        with open ('temp_data\\'+file_name,'r',encoding = 'utf-8') as csvfile:
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
    with open('temp_data\\'+'idf_i_value.csv', 'w', newline='', encoding='utf-8-sig') as file:
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
    with open('temp_data\\'+file_name, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 去除表頭
        for row in csvreader:
            for i, count in enumerate(row):
                if i == 0:
                    continue
                if int(count) > 0:
                    idf_dict[word_list[i-1]] += 1
    with open('temp_data\\'+'idf_i_value.csv', 'w', newline='', encoding='utf-8-sig') as file:
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
    df = pd.read_csv('temp_data\\'+file_name)
    i_list = []
    i_list = df['Value'].tolist()
    for index in range(len(i_list)):
        if(i_list[index] == 0):
            i_list[index] = 1
    return i_list

def caculat_idf(idf_i_list,d):
    with open('temp_data\\'+'idf_score.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        for i in idf_i_list:
            temp = 0
            temp = math.log(d/i) #以10為底，以(D/i)取log
            writer.writerow((temp,))

#-----以下函式功能：將每個文本的每個字進行tf-idf運算，生成tf-idf矩陣
def read_data(file_1,file_2):
    # file_1 為：content_total_word_number.csv , file_2 為：idf_score.csv
    with open('temp_data\\'+file_1, 'r', encoding='utf-8-sig') as readfile_1:
        total_word_number = readfile_1.readlines()
    with open('temp_data\\'+file_2, 'r', encoding='utf-8-sig') as readfile_2:
        idf_score = readfile_2.readlines()
    return total_word_number,idf_score

def tf_idf_convert(filename, total_word_number, idf_score):
    data = pd.read_csv('temp_data\\'+filename)
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
    data.to_csv('temp_data\\'+'tf_idf_matrix.csv', encoding = 'utf-8-sig', index='true')

def content_total_word(content_list):
    # 計算斷詞總共分為幾個字詞
    for text in content_list:
        total_number = 0
        words = jieba.cut(text[0])
        total_number = str(len(list(words)))
        with open('temp_data\\'+'content_total_word_number.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow((total_number,))

def concat_csv(file_1,file_2):
    # 將 tf-idf矩陣以及label文檔進行concat
    csv1_df = pd.read_csv('temp_data\\'+file_1, header=None ,low_memory=False)
    csv2_df = pd.read_csv(file_2, usecols=[1],header=None, low_memory=False)
    result_df = pd.concat([csv1_df, csv2_df], axis=1)
    result_df.to_csv('temp_data\\'+'result.csv', encoding = 'utf-8-sig', index=False, header=None) #寫成csv，file_1需輸入tf_idf_matrix；file_2需輸入具有label的原始文件
    return result_df

def main():
    # # Setting
    file_name = input("請輸入檔案名稱(如:'環保創意評價(1332) (final).csv')：")
    chunk_size = 9999999 # chunk_size 表示每寫入幾個字詞，就換到下一個csv 檔
    total_content_number = len(readfile(file_name)) # 總樣本數

    # # Variable
    total_word_number = []
    idf_score = []
    content_list = []

    # # Main function
    os.mkdir('temp_data') # 創建新資料夾儲存檔案

    to_word_vector(file_name,chunk_size) # 將原始資料轉換成詞頻矩陣
    print("tokenize_0.csv 成功輸出")

    word_list = read_word("tokenize_0.csv") # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list
    count_idf("tokenize_0.csv",word_list) # 計算i值，產生i值之csv檔
    print("idf_i_value.csv 成功輸出")

    idf_i_list = read_i_list("idf_i_value.csv") # 傳入檔名後，讀取csv 檔，並回傳存有每個row 資料的list 
    caculat_idf(idf_i_list,total_content_number) # 計算每idf之值，並將其計算結果存為csv
    print("idf_score.csv 成功輸出")

    content_list = readfile(file_name)
    content_total_word(content_list) # 計算斷詞總共分為幾個字詞
    print("content_total_word_number.csv 成功輸出")

    list_tuple = read_data('content_total_word_number.csv','idf_score.csv')
    total_word_number = list(map(float,list_tuple[0]))
    idf_score = list(map(float,list_tuple[1]))
    tf_idf_convert('tokenize_0.csv',total_word_number,idf_score) # 將矩陣進行tf-idf轉換    
    print("tf_idf_matrix.csv 成功輸出")

    concat_csv('tf_idf_matrix.csv', file_name) # 將 tf-idf矩陣以及label文檔進行concat
    print("result.csv 成功輸出")

# Main program
if __name__ == '__main__':
    main()
