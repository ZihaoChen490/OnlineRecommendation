from deepctr.layers.utils import  combined_dnn_input
from deepctr.feature_column import DenseFeat, build_input_features, create_embedding_matrix,SparseFeat, VarLenSparseFeat
from deepctr.layers.core import PredictionLayer, DNN
from tensorflow.python.keras.models import Model
from tensorflow.keras.utils import plot_model
from deepmatch.inputs import input_from_feature_columns
from deepmatch.layers.core import Similarity
from deepmatch.models import *
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import math

def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='tanh', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0.1, init_std=0.0001, seed=1024, metric='cos'):
    print(item_feature_columns)
    print(user_feature_columns)
    print(type(item_feature_columns))
    print(type(user_feature_columns))

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                     seed,     seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, None,seed )(user_dnn_input)
    print('type',type(item_dnn_input))
    print(type(dnn_use_bn))
    print(type(item_dnn_hidden_units))
    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,None, seed)(item_dnn_input)

    score = Similarity(type=metric)([user_dnn_out, item_dnn_out])

    output = PredictionLayer("binary", False)(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model

def index(dataframe):
    res=[]
    index=1
    for i in range(len(dataframe)):
        if dataframe[i] not in res:
            res.append(dataframe[i])
            dataframe[i]=index
            index+=1
        else:
            dataframe[i]=res.index(dataframe[i])+1
    return dataframe

def minmax(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

def generate_sample(uid_data,item_data,uid_item):
    cnt=0
    list_res=[]
    uid_list=uid_data.values.tolist()
    item_list=item_data.values.tolist()
    lis=uid_item.values.tolist()
    uid_item_list=[[int(lis[i][0]),int(lis[i][1])] for i in range(len(uid_item))]
    for i in range(len(uid_data)):
        print(i)
        for j in range(len(item_data)):
            res=uid_list[i]+item_list[j]
            if [res[0],res[5]] in uid_item_list:
                list_res.append(res+[1])
            else:
                list_res.append(res+[0])
    df=pd.DataFrame(list_res)
    df.columns=uid_data.columns.tolist()+item_data.columns.tolist()+['label']
    print(df.columns)
    print(df)
    return df

uid_data = pd.read_csv("D:/Dssm/Dssm/uid_info.txt", sep="\t", header=None,encoding='gbk')
uid_data.columns = ["user_id", "user_gender", "age", "province", "risk_level"]
uid_data.head()
index(uid_data['province'])
uid_data = shuffle(uid_data)

item_data = pd.read_csv("D:/Dssm/Dssm/item_info.txt", sep="\t", header=None,encoding='gbk')
item_data.columns = ["item_id", "rise_fall_rate","fund_style_attributes", "investment_style", 
"fund_establish_date","fund_manager","officeDate","fund_net_growth_rate","ranking","company_id","item_gender",
"education_level"]
item_data.head()
index(item_data['investment_style'])
index(item_data['fund_manager'])
index(item_data['investment_style'])
index(item_data['company_id'])
item_data = shuffle(item_data)

print(item_data)
uid_item_data = pd.read_csv("D:/Dssm/Dssm/uid_item.txt", sep="\t", header=None,encoding='gbk')
uid_item_data.columns = ["user_id", "item_id", "stock_share_holdings"]
uid_item_data.head()
uid_item_data = shuffle(uid_item_data)

item_data["rise_fall_rate"]=minmax(item_data["rise_fall_rate"])
item_data["fund_establish_date"]=minmax(item_data["fund_establish_date"])
item_data["officeDate"]=minmax(item_data["officeDate"])

samples_data=generate_sample(uid_data,item_data,uid_item_data)
index(samples_data['item_id'])

X = samples_data[["user_id", "user_gender", "age", "province", "risk_level", "item_id", "rise_fall_rate",
"fund_style_attributes","investment_style","fund_establish_date","fund_manager","officeDate","fund_net_growth_rate",
"company_id","ranking","item_gender","education_level"]]
y = samples_data["label"]
train_model_input = {"user_id": np.array(samples_data["user_id"]),
                         "user_gender": np.array(samples_data["user_gender"]),
                         "age": np.array(samples_data["age"]),
                         "province": np.array(samples_data["province"]),
                         "risk_level": np.array(samples_data["risk_level"]),
                         
                         "item_id": np.array(samples_data["item_id"]),
                         "rise_fall_rate": np.array(samples_data["rise_fall_rate"]),
                         "fund_style_attributes": np.array(samples_data["fund_style_attributes"]),
                         "investment_style": np.array(samples_data["investment_style"]),
                         "fund_establish_date": np.array(samples_data["fund_establish_date"]),
                         "fund_manager": np.array(samples_data["fund_manager"]),
                         "officeDate": np.array(samples_data["officeDate"]),
                         "fund_net_growth_rate": np.array(samples_data["fund_net_growth_rate"]),
                         "company_id": np.array(samples_data["company_id"]),
                         "ranking": np.array([[int(i) for i in l.split('/')] for l in samples_data["ranking"]]),
                         "item_gender": np.array(samples_data["item_gender"]),
                         "education_level": np.array(samples_data["education_level"])}
train_label = np.array(samples_data["label"])
embedding_dim = 32
SEQ_LEN = 50
user_feature_columns = [SparseFeat('user_id', max(samples_data["user_id"]) + 1, embedding_dim),
                            SparseFeat("user_gender", max(samples_data["user_gender"]) + 1, embedding_dim),
                            SparseFeat("age", max(samples_data["age"]) + 1, embedding_dim),
                            SparseFeat("province", max(samples_data["province"]) + 1, embedding_dim),
                            SparseFeat("risk_level", max(samples_data["risk_level"]) + 1, embedding_dim)]

item_feature_columns = [SparseFeat('item_id', max(samples_data["item_id"]) + 1, embedding_dim),
                        SparseFeat('fund_style_attributes', max(samples_data["fund_style_attributes"]) + 1, embedding_dim),
                        DenseFeat('rise_fall_rate',dimension=1,dtype=float),
                        SparseFeat('investment_style', max(samples_data["investment_style"]) + 1, embedding_dim),
                        DenseFeat('fund_establish_date', dimension=1,dtype=float),
                        SparseFeat('fund_manager', max(samples_data["fund_manager"]) + 1, embedding_dim),
                        DenseFeat('officeDate', dimension=1,dtype=float),
                        DenseFeat('fund_net_growth_rate', dimension=1,dtype=float),
                        DenseFeat('ranking', dimension=2,dtype=float),
                        SparseFeat('company_id', max(samples_data["company_id"]) + 1, embedding_dim),
                        SparseFeat("item_gender", max(samples_data["item_gender"]) + 1, embedding_dim),
                        SparseFeat("education_level", max(samples_data["education_level"]) + 1, embedding_dim) ]
model = DSSM(user_feature_columns, item_feature_columns)
model.compile(optimizer='adagrad', loss="binary_crossentropy", metrics=['accuracy'])
history = model.fit(train_model_input, train_label,
                    batch_size=64, epochs=10, verbose=1, validation_split=0.2, )
model._layers = [layer for layer in model.layers if not isinstance(layer, dict)]
model.summary()
plot_model(model, to_file='DSSM_model.png', show_shapes=True)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
print(plt.show())