import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

target_train = pd.read_csv("../data/预测目标.训练集.csv")
target_train.dropna(inplace=True)
target_train.reset_index(inplace=True)

##找到并记录每个月最后一天
a = target_train['biz_date'].str.split('-')
index_list = []
month_list = []
# 记录的是每个月最后一天的index
for i,s in enumerate(a):
    if i<len(a)-1 and a[i][1] != a[i+1][1]:
        index_list.append(i)
        month_list.append(str(s[0]) + "-" + str(s[1]))
        print(str(i) + ": " + str(s[0]) + "-" + str(s[1]))
        
        
## 定义需要的函数
def data_organizer_array(X_list):
    for i in range(len(X_list)):
        temp_X = np.array(X_list[i])
        n_input = temp_X.shape[0] * temp_X.shape[1]
        temp_X = temp_X.reshape((1, n_input))
        if i == 0:
            X_data = temp_X
        else:
            X_data = np.vstack((X_data,temp_X))
    return X_data

# MLP模型
def multi_parallel_output_model(n_input, n_output, X, y, epochs_num,n_neuron):
    model = Sequential()
    model.add(Dense(n_neuron, activation='relu', input_dim=n_input))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=epochs_num, verbose=0)
    return model


##主函数
def main(n_neuron1, n_neuron2, n_neuron3, var_start, var_end):
    print("Model 1:")
    # train_X
    temp_X_list = []
    for i in range(1,len(index_list)-1):
        index_end = index_list[i]
        temp_X = target_train.loc[index_end-20:index_end,var_start:var_end]
        temp_X = np.array(temp_X)
        temp_X_list.append(temp_X)
    train_X = data_organizer_array(temp_X_list)
    print(train_X.shape)

    # train_Y
    temp_Y_list = []
    for i in range(1,len(index_list)-1):
        index_start = index_list[i] + 1
        temp_Y = target_train.loc[index_start:index_start+4,var_start:var_end]
        temp_Y = np.array(temp_Y)
        temp_Y_list.append(temp_Y)
    train_Y = data_organizer_array(temp_Y_list)
    print(train_Y.shape)

    # test_X
    index_end = index_list[-1]
    X_last = target_train.loc[index_end-20:index_end,var_start:var_end]
    X_last = np.array(X_last)
    test_X = X_last.reshape(1,-1)
    print(test_X.shape)

    # test_Y
    original_index_start = target_train['index'][index_list[-1]] + 1
    index_start = np.where(target_train['index']==original_index_start)[0][0]
    Y_last = target_train.loc[index_start:index_start+4,var_start:var_end]
    Y_last = np.array(Y_last)
    test_Y = Y_last.reshape(1,-1)
    print(test_Y.shape)

    # 模型一
    # 建立模型
    n_input = train_X.shape[1]
    n_output = train_Y.shape[1]
    epochs_num = 2000
    # n_neuron1 = 300
    model = multi_parallel_output_model(n_input, n_output, train_X, train_Y, epochs_num,n_neuron1)
    # 预测并算分
    y_pre = model.predict(test_X,verbose=0)
    score1 = np.mean(abs((y_pre-test_Y)/test_Y))
    # print("score1: " + str(score1))
    # 新建列表，储存每部预测结果
    pre_list = []
    pre_list.append(y_pre)
    
    print("Model 2:")
    # train_X_update
    vars_num = int(train_X.shape[1]/21)
    train_Y_est = model.predict(train_X,verbose=0)
    train_Y_est_list = []
    for i in range(len(train_Y_est)):
        a = train_Y_est[i]
        a = a.reshape(5,vars_num)
        train_Y_est_list.append(a)
    for i,array in enumerate(temp_X_list):
        add_X = train_Y_est_list[i]
        new_X = np.vstack((array,add_X))
        new_X = new_X.reshape(1,-1)
        if i > 0 :
            train_X_update = np.vstack((train_X_update,new_X))
        else:
            train_X_update = new_X
    print(train_X_update.shape)

    # train_Y_update
    temp_Y_list_update = []
    for i in range(1,len(index_list)-1):
        index_start = index_list[i] + 6
        temp_Y = target_train.loc[index_start:index_start+5,var_start:var_end]
        temp_Y = np.array(temp_Y)
        temp_Y_list_update.append(temp_Y)
    train_Y_update = data_organizer_array(temp_Y_list_update)
    print(train_Y_update.shape)

    # test_X_update
    test_X_update = np.vstack((test_X.reshape(21,vars_num),y_pre.reshape(5,vars_num)))
    test_X_update = test_X_update.reshape(1,26*vars_num)
    print(test_X_update.shape)

    # test_Y_update
    original_index_start = target_train['index'][index_list[-1]] + 1
    index_start = np.where(target_train['index']==original_index_start)[0][0]
    Y_last = target_train.loc[index_start+5:index_start+10,var_start:var_end]
    Y_last = np.array(Y_last)
    test_Y_update = Y_last.reshape(1,-1)
    print(test_Y_update.shape)

    # 模型二
    # 建立模型
    n_input_update = train_X_update.shape[1]
    n_output_update = train_Y_update.shape[1]
    epochs_num = 2000
    # n_neuron2 = 400
    model_update = multi_parallel_output_model(n_input_update, n_output_update, train_X_update, train_Y_update, epochs_num,n_neuron2)
    # 进行预测并算分
    test_X_update = test_X_update.astype('float32')
    y_pre_update = model_update.predict(test_X_update,verbose=0)
    score2 = np.mean(abs((y_pre_update-test_Y_update)/test_Y_update))
    # print("score2: " +str(score2))
    pre_list.append(y_pre_update)
    
    print("Model 3:")
    # train_X_update_final
    train_Y_update_est = model_update.predict(train_X_update,verbose=0)
    train_Y_est_update_list = []
    for i in range(len(train_Y_update_est)):
        a = train_Y_update_est[i]
        a = a.reshape(6,vars_num)
        train_Y_est_update_list.append(a)
    for i,array in enumerate(temp_X_list):
        add_X1 = train_Y_est_list[i]
    #     print(add_X1.shape)
        add_X2 = train_Y_est_update_list[i]
    #     print(add_X2.shape)
    #     print(array.shape)
        new_X = np.vstack((array,add_X1,add_X2))
        new_X = new_X.reshape(1,-1)
    #     print(new_X.shape)
        print
        if i > 0 :
            train_X_update_final = np.vstack((train_X_update_final,new_X))
        else:
            train_X_update_final = new_X
    print(train_X_update_final.shape)

    # train_Y_update_final
    temp_Y_list_update_final = []
    for i in range(1,len(index_list)-1):
        index_start = index_list[i] + 12
        temp_Y = target_train.loc[index_start:index_start+7,var_start:var_end]
        temp_Y = np.array(temp_Y)
        temp_Y_list_update_final.append(temp_Y)
    train_Y_update_final = data_organizer_array(temp_Y_list_update_final)
    print(train_Y_update_final.shape)

    # test_X_update_final
    test_X_update_final = np.vstack((test_X.reshape(21,vars_num),y_pre.reshape(5,vars_num),y_pre_update.reshape(6,vars_num)))
    test_X_update_final = test_X_update_final.reshape(1,32*vars_num)
    print(test_X_update_final.shape)

    # test_Y_update_final
    original_index_start = target_train['index'][index_list[-1]] + 1
    index_start = np.where(target_train['index']==original_index_start)[0][0]
    Y_last = target_train.loc[index_start+11:index_start+18,var_start:var_end]
    Y_last = np.array(Y_last)
    test_Y_update_final = Y_last.reshape(1,-1)
    print(test_Y_update_final.shape)

    # 模型三
    # 建立模型
    n_input_update_final = train_X_update_final.shape[1]
    n_output_update_final = train_Y_update_final.shape[1]
    epochs_num = 2000
    # n_neuron3 = 500
    model_update_final = multi_parallel_output_model(n_input_update_final, n_output_update_final, train_X_update_final, train_Y_update_final, epochs_num,n_neuron3)
    # 预测并算分
    test_X_update_final = test_X_update_final.astype('float32')
    y_pre_update_final = model_update_final.predict(test_X_update_final,verbose=0)
    score3 = np.mean(abs((y_pre_update_final-test_Y_update_final)/test_Y_update_final))
    # print("score3: "+ str(score3))
    pre_list.append(y_pre_update_final)

    # 返回预测结果的array
    for i,p in enumerate(pre_list):
        n_row = int(p.shape[1]/vars_num)
        p = p.reshape(n_row,vars_num)
        if i > 0:
            result = np.vstack((result,p))
        else:
            result = p
    print(result.shape)
    return [result, model, model_update, model_update_final]

##参数
# 国债
para_treasury = {'var_start':'treasury_bond_rate_1m', 'var_end':'treasury_bond_rate_10y'}
# 国开债
para_cdb = {'var_start':'cdb_rate_6m', 'var_end':'cdb_rate_10y'}
# 地方政府债
para_loc = {'var_start':'loc_rate_6m', 'var_end':'loc_rate_10y'}
# 城投债
para_cdi = {'var_start':'cdi_rate_6m', 'var_end':'cdi_rate_10y'}
# 企业债
para_com = {'var_start':'com_rate_1m', 'var_end':'com_rate_10y'}
para_list = [para_treasury, para_cdb, para_loc, para_cdi, para_com]
# 四月真实数据
original_index_start = target_train['index'][index_list[-1]] + 1
apirl_index_start = np.where(target_train['index']==original_index_start)[0][0]
true_rate = target_train.loc[apirl_index_start:apirl_index_start+18,'treasury_bond_rate_1m':'com_rate_10y']
true_rate = np.array(true_rate)
print(true_rate.shape)
def cal_score(n_neuron1, n_neuron2, n_neuron3):
    models_list = []
    for i, para_dic in enumerate(para_list):
        pre_result,model1,model2,model3 = main(n_neuron1, n_neuron2, n_neuron3,**para_dic)
        if i > 0:
            final_result = np.hstack((final_result, pre_result))
        else:
            final_result = pre_result
        models_list.append([model1,model2,model3])
    delta_relative = abs(final_result - true_rate) / true_rate
    delta_relative = np.mean(delta_relative,1)
    weight = [0.5/5]*5 + [0.25/6]*6 + [0.25/12]*8
    final_score = delta_relative.dot(np.array(weight))
    print("")
    print(final_score)
    return models_list

##最终预测
def main_pre(model1, model2, model3, var_start, var_end):
    # pre_X
    pre_index_start = list(target_train.index)[-21]
    pre_X = target_train.loc[pre_index_start:pre_index_start+20,var_start:var_end]
    pre_X = np.array(pre_X)
    vars_num = pre_X.shape[1]
    pre_X = pre_X.reshape(1,-1)
    print(pre_X.shape)
    # y_pre
    y_pre = model1.predict(pre_X,verbose=0)
    pre_list = []
    pre_list.append(y_pre)
    
    # pre_X_update
    pre_X_update = np.vstack((pre_X.reshape(21,vars_num),y_pre.reshape(5,vars_num)))
    pre_X_update = pre_X_update.reshape(1,-1)
    print(pre_X_update.shape)

    # y_pre_update
    y_pre_update = model2.predict(pre_X_update,verbose=0)
    pre_list.append(y_pre_update)

    # pre_X_update_final
    pre_X_update_final = np.vstack((pre_X.reshape(21,vars_num),y_pre.reshape(5,vars_num),y_pre_update.reshape(6,vars_num)))
    pre_X_update_final = pre_X_update_final.reshape(1,-1)
    print(pre_X_update_final.shape)

    # y_pre_update_final
    y_pre_update_final = model3.predict(pre_X_update_final,verbose=0)
    pre_list.append(y_pre_update_final)
    
    # 返回预测结果的array
    for i,p in enumerate(pre_list):
        n_row = int(p.shape[1]/vars_num)
        p = p.reshape(n_row,vars_num)
        if i > 0:
            result = np.vstack((result,p))
        else:
            result = p
    print(result.shape)
    
    # 加上4月倒数两天的数据，构成最后6月4天的预测X
    apirl_last_index = list(target_train.index)[-1]
    add_X = target_train.loc[apirl_last_index-1:apirl_last_index, var_start:var_end]
    add_X = np.array(add_X)
    may_X = np.vstack((add_X, result))
    may_X = may_X.reshape(1,-1)
    # 预测6月前5个交易日
    y_june = model1.predict(may_X,verbose=0)
    y_june = y_june.reshape(5,vars_num)
    
    # 最终结果
    final_ersult = np.vstack((result,y_june))[:-1]
    print(final_ersult.shape)
    
    return final_ersult

## 重复5次训练取平均
para_list = [para_treasury, para_cdb, para_loc, para_cdi, para_com]
june_pre_list = []
repeats = 5
for r in range(repeats):
    models_list_final = cal_score(100,200,300)
    for i,para in enumerate(para_list):
        models = models_list_final[i]
        june_pre = main_pre(models[0], models[1], models[2], **para)
        if i > 0:
            pre_merge = np.hstack((pre_merge,june_pre))
        else:
            pre_merge = june_pre
    june_pre_list.append(pre_merge)
    
## 最终导出
final_pre = (june_pre_list[0] + june_pre_list[1] + june_pre_list[2] + june_pre_list[3] + june_pre_list[4])/5
final_pre = pd.DataFrame(final_pre)
final_pre.columns = target_train.columns[2:]
final_pre.to_csv('../output/prediction.csv')
print("导出文件prediction.csv完毕")