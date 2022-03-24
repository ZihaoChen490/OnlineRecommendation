import tensorflow as tf
from tensorflow.keras import layers
from layers import AUGRU
from activations import Dice
import pandas as pd
from model import DIEN
import data_reader as data_reader
import numpy as np
import matplotlib.pyplot as plt
def train_one_step(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, optimizer, model, alpha, loss_metric,batch_size):
    with tf.GradientTape() as tape:
        output, logit, aux_loss = model(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list,batch_size)
        target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=tf.cast(label, dtype=tf.float32)))
        final_loss = target_loss + alpha * aux_loss
        print("[Train Step] aux_loss=" + str(aux_loss.numpy()) + ", target_loss=" + str(target_loss.numpy()) + ", final_loss=" + str(final_loss.numpy()))
    gradient = tape.gradient(final_loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))
    loss_metric(final_loss)
    return aux_loss, target_loss, final_loss

def get_test_loss(model,user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, batch):
    output, logit = model(
        user_profile_dict, 
        user_profile_list, 
        click_behavior_dict, 
        target_item_dict, 
        noclick_behavior_dict,
        user_behavior_list, 
        batch
    )
    final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=tf.cast(label, dtype=tf.float32)))
    print("final_loss=" + str(final_loss))
    return final_loss.numpy()

def get_loss_fig(train_loss, test_loss):
    loss_list = ["final_loss"]
    color_list = ["r", "b"]
    plt.figure()
    cnt = 0
    for k in loss_list:
        loss = train_loss[k]
        step = list(np.arange(len(loss)))
        plt.plot(step,loss,color_list[cnt]+"-",label="train_" + k, linestyle="--")
        cnt += 1
    cnt = 0
    for k in loss_list:
        loss = test_loss[k]
        step = list(np.arange(len(loss)))
        plt.plot(step,loss,color_list[cnt],label="test_" + k)
        cnt += 1
    plt.title("Loss")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    clear_output()
    mkdir("./loss/" + model_name)
    plt.savefig("./loss/" + model_name + "/loss.png")
    clear_output()
    plt.show()

def main():
    train_data, test_data, embedding_count = data_reader.get_data()
    embedding_features_list = data_reader.get_embedding_features_list()
    user_behavior_features = data_reader.get_user_behavior_features()
    embedding_count_dict = data_reader.get_embedding_count_dict(embedding_features_list, embedding_count)
    embedding_dim_dict = data_reader.get_embedding_dim_dict(embedding_features_list)
    #for features in embedding_count_dict:
        #print('features is', features)
        #print(embedding_count_dict[features])
    model = DIEN(embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features)
    min_batch = 0
    batch = 2000
    print(len(train_data))
    click,banner_pos, id,site_id, site_domain, site_category, app_id, app_domain, app_category, device_id, device_model, device_type, device_conn_type,  hour, C1, C14,C15, C16,C17, C18,C19, C20,C21, min_batch,reshape_len = data_reader.get_batch_data(train_data, min_batch, batch = batch)
    user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data( banner_pos, id,site_id, site_domain, site_category, app_id, app_domain, app_category, device_id, device_model, device_type, device_conn_type,  hour, C1, C14,C15, C16,C17, C18,C19, C20,C21, min_batch) 
    log_path = "./train_log/"
    train_summary_writer = tf.summary.create_file_writer(log_path)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_metric = tf.keras.metrics.Sum()
    auc_metric = tf.keras.metrics.AUC()
    alpha = 1
    epochs = 1

    for epoch in range(epochs):
        print('epoch now is',epoch,'with',epochs)
        min_batch = 0
        for i in range(int(len(train_data) / batch)//100):
            if i%100==0:
                print(i,len(train_data)//batch)
            label,banner_pos, id,site_id, site_domain, site_category, app_id, app_domain, app_category, device_id, device_model, device_type, device_conn_type,  hour, C1, C14,C15, C16,C17, C18,C19, C20,C21, min_batch,reshape_len = data_reader.get_batch_data(train_data, min_batch, batch = batch)
            user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data( banner_pos,id,site_id, site_domain, site_category, app_id, app_domain, app_category, device_id, device_model, device_type, device_conn_type,hour, C1, C14,C15, C16,C17, C18,C19, C20,C21, min_batch)
            label=tf.constant(label,dtype=tf.float32)
            train_one_step(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, optimizer, model, alpha, loss_metric,batch)
    #model.save_weights('./model/weight_save.h5')
    #model=tf.keras.models.load_model('./model/weight_save.h5')
    get_test_loss(model,user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, batch)
    model.summary()


if __name__ == "__main__":
    print(tf.__version__)
    print("GPU Available: ", tf.test.is_gpu_available())
    main()

