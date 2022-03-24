import pandas as pd
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors,word2vec,Word2Vec
from torch import int16

def get_embedding_features_list():
    embedding_features_list = ["banner_pos","id","hour", "C1", "site_id", "site_domain", "site_category", "app_id", 
                           "app_domain", "app_category","device_id", "device_model", "device_type", 
                           "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19","C20", "C21"]
    return embedding_features_list
    
def get_user_behavior_features():
    user_behavior_features = ["site_id", "site_domain", "site_category","hour",
    "id","banner_pos","C1", "C14","C15","C16","C17", "C18","C19","C20", "C21"]
    return user_behavior_features

def get_embedding_count(feature, embedding_count):
    if isinstance(embedding_count[feature].max(),str):
        if int(embedding_count[feature].max(),16)>1e12:
            return int(np.array(list(map(decode,embedding_count[feature].values))).astype(np.float).max())+1
        return int(np.array(list(map(decode,embedding_count[feature].values))).astype(np.float).max())+1
    else:
        res=np.array(list(map(decode,list(map(str,embedding_count[feature]-embedding_count[feature].min()+1))))).astype(np.float)
        res=res/res.max()
        return int(res.max())+1

def get_embedding_count_dict(embedding_features_list, embedding_count):
    embedding_count_dict = dict()
    #print(embedding_count)
    for feature in embedding_features_list:
        embedding_count_dict[feature] = get_embedding_count(feature, embedding_count)    
    embedding_count_dict["id"]=int(len(embedding_count["id"]))
    #embedding_count_click = pd.read_csv("./data/sampleSubmission.csv")
    return embedding_count_dict

def get_embedding_dim_dict(embedding_features_list):
    embedding_dim_dict = dict()
    for feature in embedding_features_list:
        embedding_dim_dict[feature] = 64
    return embedding_dim_dict

def get_data():
    train_data = pd.read_csv("./data/train.csv")
    train_data = train_data.fillna(0)
    valid_data=train_data.sample(frac=0.3,replace=False)
    index = valid_data.index.tolist()
    train_data.drop(index,inplace=True)
    train_data.sort_index(inplace=True)
    valid_data.sort_index(inplace=True)
    #train_data = train_data[train_data["click"] != 0]
    #train_data = train_data[train_data["banner_pos"] != 0]
    test_data = pd.read_csv("./data/test.csv")
    test_data = test_data.fillna(0)
    #test_data = test_data[test_data["click"] != 0]
    #test_data = test_data[test_data["banner_pos"] != 0]
    embedding_count={}
    for features in train_data:
        embedding_count[features]=train_data[features]
    return train_data, valid_data,test_data, embedding_count

def get_normal_data(data, col):
    if isinstance(data[col].max(),str):
        if int(data[col].max(),16)>1e12:
            return np.array(list(map(decode,data[col].values))).astype(np.float)
        return np.array(list(map(decode,data[col].values))).astype(np.float)
    else:
        res=np.array(list(map(decode,list(map(str,data[col]-data[col].min()+1))))).astype(np.float)
        res=res/res.max()
        return res

def decode(text):
    Offset=0
    for I in text:
        string = ""
        ASCII = ord(I)
        Result = ASCII + Offset
    return np.array(Result).astype(np.float)

def get_sequence_data(data, col):
    rst = []
    max_length = 0
    for i in data[col].values:
        temp = len(list(map(eval,i[1:-1].split(","))))
        if temp > max_length:
            max_length = temp

    for i in data[col].values:
        temp = list(map(eval,i[1:-1].split(",")))
        padding = np.zeros(max_length - len(temp))
        rst.append(list(np.append(np.array(temp), padding)))
    return rst

def get_length(data, col):
    rst = []
    for i in data[col].values:
        temp = len(list(map(eval,i[1:-1].split(","))))
        rst.append(temp)
    return rst

def convert_tensor(data):
    return tf.convert_to_tensor(data)

def get_batch_data(data, min_batch, batch=100):
    # batch_data = None
    # if min_batch + batch <= len(data):
    #     batch_data = data.loc[min_batch:min_batch + batch - 1]
    # else:
    #     batch_data = data.loc[min_batch:]
    batch_data = data.sample(n=batch)
    id =get_normal_data(batch_data, "id")
    click = batch_data["click"]
    banner_pos = get_normal_data(batch_data, "banner_pos")

    hour = get_normal_data(batch_data, "hour")
    C1=get_normal_data(batch_data, "C1")
    C14=get_normal_data(batch_data, "C14")
    C15=get_normal_data(batch_data, "C15")
    C16=get_normal_data(batch_data, "C16")
    C17=get_normal_data(batch_data, "C17")
    C18=get_normal_data(batch_data, "C18")
    C19=get_normal_data(batch_data, "C19")
    C20=get_normal_data(batch_data, "C20")
    C21=get_normal_data(batch_data, "C21")
    #no_click = get_normal_data(batch_data, "guide_dien_final_train_data.nonclk")
    #label = [click, no_click]
    label = click
    site_id = get_normal_data(batch_data, "site_id")
    site_domain =  get_normal_data(batch_data, "site_domain")
    site_category = get_normal_data(batch_data, "site_category")
    app_id =  get_normal_data(batch_data, "app_id")
    app_domain = get_normal_data(batch_data, "app_domain")
    app_category = get_normal_data(batch_data, "app_category")
    device_id = get_normal_data(batch_data, "device_id")
    device_model = get_normal_data(batch_data, "device_model")
    device_type = get_normal_data(batch_data, "device_type")
    device_conn_type = get_normal_data(batch_data, "device_conn_type")
    reshape_len = convert_tensor(label).numpy().shape[0]
    #print('reshape_len is',reshape_len)
    #model=word2vec.LineSentence(id)
    return tf.one_hot(click, 2), convert_tensor(banner_pos),convert_tensor(id), convert_tensor(site_id), convert_tensor(site_domain), convert_tensor(site_category), convert_tensor(app_id), convert_tensor(app_domain), convert_tensor(app_category), convert_tensor(device_id), convert_tensor(device_model), convert_tensor(device_type), convert_tensor(device_conn_type),  convert_tensor(hour), convert_tensor(C1), convert_tensor(C14),convert_tensor(C15), convert_tensor(C16),convert_tensor(C17), convert_tensor(C18),convert_tensor(C19), convert_tensor(C20),convert_tensor(C21), min_batch + batch,reshape_len

def get_test_batch(data,min_batch,batch):
    min_batch = 0
    batch_data = data.sample(n=batch)
    id =get_normal_data(batch_data, "id")
    banner_pos = get_normal_data(batch_data, "banner_pos")

    hour =get_normal_data(batch_data, "banner_pos")
    C1=get_normal_data(batch_data, "C1")
    C14=get_normal_data(batch_data, "C14")
    C15=get_normal_data(batch_data, "C15")
    C16=get_normal_data(batch_data, "C16")
    C17=get_normal_data(batch_data, "C17")
    C18=get_normal_data(batch_data, "C18")
    C19=get_normal_data(batch_data, "C19")
    C20=get_normal_data(batch_data, "C20")
    C21=get_normal_data(batch_data, "C21")
    #no_click = get_normal_data(batch_data, "guide_dien_final_train_data.nonclk")
    #label = [click, no_click]
    site_id = get_normal_data(batch_data, "site_id")
    site_domain =  get_normal_data(batch_data, "site_domain")
    site_category = get_normal_data(batch_data, "site_category")
    app_id =  get_normal_data(batch_data, "app_id")
    app_domain = get_normal_data(batch_data, "app_domain")
    app_category = get_normal_data(batch_data, "app_category")
    device_id = get_normal_data(batch_data, "device_id")
    device_model = get_normal_data(batch_data, "device_model")
    device_type = get_normal_data(batch_data, "device_type")
    device_conn_type = get_normal_data(batch_data, "device_conn_type")
    #print('reshape_len is',reshape_len)
    return convert_tensor(banner_pos),convert_tensor(id), convert_tensor(site_id), convert_tensor(site_domain), convert_tensor(site_category), convert_tensor(app_id), convert_tensor(app_domain), convert_tensor(app_category), convert_tensor(device_id), convert_tensor(device_model), convert_tensor(device_type), convert_tensor(device_conn_type),  convert_tensor(hour), convert_tensor(C1), convert_tensor(C14),convert_tensor(C15), convert_tensor(C16),convert_tensor(C17), convert_tensor(C18),convert_tensor(C19), convert_tensor(C20),convert_tensor(C21), min_batch + batch


def get_train_data(banner_pos,id,site_id, site_domain, site_category, app_id, app_domain, app_category, 
device_id, device_model, device_type, device_conn_type, 
 hour, C1, C14,C15, C16,C17, C18,C19, C20,C21, min_batch):
    user_profile_dict = {
        "app_id": app_id,
        "app_domain": app_domain,
        "app_category": app_category,
        "device_id": device_id,
        "device_model": device_model,
        "device_type": device_type,
        "device_conn_type": device_conn_type,
        "hour": hour,

    }
    user_profile_list = ["app_id", "app_domain", "app_category", "device_id", 
    "device_model", "device_type", "device_conn_type", "hour" ]
    user_behavior_list = ["site_id", "site_domain", "site_category","hour",
    "id","banner_pos","C1", "C14","C15","C16","C17", "C18","C19","C20", "C21"    
    ]

    click_behavior_dict = {
        "site_id": site_id,
        "site_domain": site_domain,
        "site_category": site_category,
         "id": id,
       "hour": hour,
        "banner_pos": banner_pos,
        "C1": C1,
        "C14": C14,
        "C15": C15,
        "C16": C16,
        "C17": C17,
        "C18": C18,
        "C19": C19,
        "C20": C20,
        "C21": C21
    }
    noclick_behavior_dict = {
        "site_id": site_id,
        "site_domain": site_domain,
        "site_category": site_category,
         "id": id,
       "hour": hour,
        "banner_pos": banner_pos,
        "C1": C1,
        "C14": C14,
        "C15": C15,
        "C16": C16,
        "C17": C17,
        "C18": C18,
        "C19": C19,
        "C20": C20,
        "C21": C21
    }
    target_item_dict = {
        "site_id": site_id,
        "site_domain": site_domain,
        "site_category": site_category,
         "id": id,
       "hour": hour,
        "banner_pos": banner_pos,
        "C1": C1,
        "C14": C14,
        "C15": C15,
        "C16": C16,
        "C17": C17,
        "C18": C18,
        "C19": C19,
        "C20": C20,
        "C21": C21
    }
    return user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict



def get_test_data(banner_pos,id,site_id, site_domain, site_category, app_id, app_domain, app_category, 
device_id, device_model, device_type, device_conn_type, 
 hour, C1, C14,C15, C16,C17, C18,C19, C20,C21, min_batch):
    user_profile_dict = {

        "app_id": app_id,
        "app_domain": app_domain,
        "app_category": app_category,
        "device_id": device_id,
        "device_model": device_model,
        "device_type": device_type,
        "device_conn_type": device_conn_type,
        "hour": hour

    }
    user_profile_list = ["app_id", "app_domain", "app_category", "device_id", 
     "device_model", "device_type", "device_conn_type", "hour" ]
    user_behavior_list = ["site_id", "site_domain", "site_category","hour",
    "id","banner_pos","C1", "C14","C15","C16","C17", "C18","C19","C20", "C21"    
    ]
    click_behavior_dict = {
        "site_id": site_id,
        "site_domain": site_domain,
        "site_category": site_category,
         "id": id,
       "hour": hour,
        "banner_pos": banner_pos,
        "C1": C1,
        "C14": C14,
        "C15": C15,
        "C16": C16,
        "C17": C17,
        "C18": C18,
        "C19": C19,
        "C20": C20,
        "C21": C21
    }
    noclick_behavior_dict = {
        "site_id": site_id,
        "site_domain": site_domain,
        "site_category": site_category,
         "id": id,
       "hour": hour,
        "banner_pos": banner_pos,
        "C1": C1,
        "C14": C14,
        "C15": C15,
        "C16": C16,
        "C17": C17,
        "C18": C18,
        "C19": C19,
        "C20": C20,
        "C21": C21
    }
    target_item_dict = {
        "site_id": site_id,
        "site_domain": site_domain,
        "site_category": site_category,
         "id": id,
       "hour": hour,
        "banner_pos": banner_pos,
        "C1": C1,
        "C14": C14,
        "C15": C15,
        "C16": C16,
        "C17": C17,
        "C18": C18,
        "C19": C19,
        "C20": C20,
        "C21": C21
    }
    return user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict
