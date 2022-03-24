import tensorflow as tf
from tensorflow.keras import layers
from layers import AUGRU,attention
from activations import Dice,dice
from loss import AuxLayer
import utils

class DIEN(tf.keras.Model):
    def __init__(self, embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation="PReLU"):
        super(DIEN, self).__init__(embedding_count_dict, embedding_dim_dict, embedding_features_list, activation)
        #Init Embedding Layer
        self.embedding_dim_dict = embedding_dim_dict
        self.embedding_count_dict = embedding_count_dict
        self.embedding_layers = dict()
        for feature in embedding_features_list:
            self.embedding_layers[feature] = layers.Embedding(embedding_count_dict[feature], embedding_dim_dict[feature])
        #Init GRU Layer
        self.user_behavior_gru = layers.GRU(self.get_GRU_input_dim(embedding_dim_dict, user_behavior_features), return_sequences=True)
        #Init Attention Layer
        self.attention_layer = layers.Softmax()
        #Init Auxiliary Layer
        self.AuxNet = AuxLayer()
        #Init AUGRU Layer
        self.user_behavior_augru = AUGRU(self.get_GRU_input_dim(embedding_dim_dict, user_behavior_features))
        #Init Fully Connection Layer
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(200, activation="relu")) 
        if activation == "Dice":
            self.fc.add(Dice())
        elif activation == "dice":
            self.fc.add(dice(200))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(80, activation="relu"))
        if activation == "Dice":
            self.fc.add(Dice()) 
        elif activation == "dice":
            self.fc.add(dice(80))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(2, activation=None))

    def get_GRU_input_dim(self, embedding_dim_dict, user_behavior_features):
        rst = 0
        for feature in user_behavior_features:
            rst += embedding_dim_dict[feature]
        return rst

    def get_emb(self, user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list):
        user_profile_feature_embedding = dict()
        for feature in user_profile_list:
            #print(feature)
            data = user_profile_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            user_profile_feature_embedding[feature] = tf.reshape(embedding_layer(data),shape=[-1,1,64])
            #print(user_profile_feature_embedding[feature].shape)

        target_item_feature_embedding = dict()
        for feature in user_behavior_list:
            #print(feature)
            data = target_item_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            target_item_feature_embedding[feature] = tf.reshape(embedding_layer(data),shape=[-1,1,64])
            #print(target_item_feature_embedding[feature].shape)

        click_behavior_embedding = dict()
        for feature in user_behavior_list:
            #print(feature)
            data = click_behavior_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            click_behavior_embedding[feature] = tf.reshape(embedding_layer(data),shape=[-1,1,64])
            #print(click_behavior_embedding[feature].shape)

        #print('sssssssssssssssssssssssssssssssssssssssssssssssss')
        # noclick_behavior_embedding = dict()
        # for feature in user_behavior_list:
        #     data = noclick_behavior_dict[feature]
        #     embedding_layer = self.embedding_layers[feature]
        #     noclick_behavior_embedding[feature] = embedding_layer(data)
        
        return utils.concat_features(user_profile_feature_embedding), utils.concat_features(target_item_feature_embedding), utils.concat_features(click_behavior_embedding)#, utils.concat_features(noclick_behavior_embedding)

    def auxiliary_loss(self, hidden_states, embedding_out):
        #print(hidden_states.shape)
        #print(embedding_out.shape)
        click_input_ = tf.concat([hidden_states, embedding_out],1)
        #print('click_inp',click_input_.shape)
        click_prop_ = self.AuxNet(click_input_)[:, 0, :]
        #print('click_prop_',click_prop_)
        click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(embedding_out)[1]])
        return tf.reduce_mean(click_loss_)
    
    def call(self, user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list,batch_size):
        """输入batch训练数据, 调用DIEN初始化后的model进行一次前向传播
        调用该函数进行一次前向传播得到output, logit, aux_loss后,在自定义的训练函数内得出target_loss与final_loss后
        使用tensorflow中的梯度计算函数通过链式法则得到各层梯度后使用自定义优化器进行一次权重更新
        变量名称：
            user_profile_dict:dict:string->Tensor格式,记录user_profile部分的所有输入特征的训练数据;
            user_profile_list:list(string)格式,记录user_profile部分的所有特征名称;
            click_behavior_dict:dict:string->Tensor格式,记录user_behavior部分所有点击输入特征的训练数据;
            noclick_behavior_dict:dict:string->Tensor格式,记录user_behavior部分所有未点击输入特征的训练数据;
            target_item_dict:dict:string->Tensor格式,记录target_item部分输入特征的训练数据;
            user_behavior_list:list(string)Tensor格式,记录user_behavior部分的所有特征名称。
        """
        #Embedding Layer
        user_profile_embedding, target_item_embedding, click_behavior_emebedding = self.get_emb(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list)
        #GRU Layer
        click_gru_emb = self.user_behavior_gru(click_behavior_emebedding)
        #print(tf.reshape(click_gru_emb,shape=[100,64,2]))
        #target_item_embedding=tf.reshape(target_item_embedding,shape=[100,64,2])
        #noclick_gru_emb = self.user_behavior_gru(noclick_behavior_emebedding)
        #Auxiliary Loss
        #print(click_behavior_emebedding.shape)
        #print(click_gru_emb.shape)
        aux_loss = self.auxiliary_loss(click_gru_emb[:, :, :], click_behavior_emebedding[:, :, :])
        #Attention Layer
        hist_attn = self.attention_layer(tf.matmul(tf.expand_dims(target_item_embedding, 1), click_gru_emb, transpose_b=True))
        #AUGRU Layer
        #print(hist_attn.shape)
        hist_attn=tf.reshape(hist_attn,shape=[batch_size,1,batch_size])
        #hist_attn=tf.expand_dims(hist_attn,1)
        #print(click_gru_emb.shape)
        augru_hidden_state = tf.zeros_like(click_gru_emb[:, 0, :])
        #print(tf.transpose(hist_attn, [2, 0, 1]).shape)
        #print(augru_hidden_state.shape)
        #print(tf.transpose(click_gru_emb, [1, 0, 2]).shape)
        for in_emb, in_att in zip(tf.transpose(click_gru_emb, [1, 0, 2]), tf.transpose(hist_attn, [2, 0, 1])):
            augru_hidden_state = self.user_behavior_augru(in_emb, augru_hidden_state, in_att)
        #print(user_profile_embedding.shape)
        #print(augru_hidden_state.shape)
        user_profile_embedding=tf.reshape(user_profile_embedding,shape=([batch_size,-1]))
        augru_hidden_state=tf.reshape(augru_hidden_state,shape=[batch_size,-1])
        join_emb = tf.concat([augru_hidden_state, user_profile_embedding], -1)
        logit = tf.squeeze(self.fc(join_emb))
        output = tf.keras.activations.softmax(logit)
        return output, logit, aux_loss

class DIN(tf.keras.Model):
    def __init__(self, embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation="PReLU"):
        super(DIN, self).__init__(embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation)
        #Init Embedding Layer
        self.embedding_dim_dict = embedding_dim_dict
        self.embedding_count_dict = embedding_count_dict
        self.embedding_layers = dict()
        for feature in embedding_features_list:

            self.embedding_layers[feature] = layers.Embedding(embedding_count_dict[feature], embedding_dim_dict[feature])
        #DIN Attention+Sum pooling
        self.hist_at = attention(utils.get_input_dim(embedding_dim_dict, user_behavior_features))
        #Init Fully Connection Layer
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(200, activation="relu")) 
        if activation == "Dice":
            self.fc.add(Dice())
        elif activation == "dice":
            self.fc.add(dice(200))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(80, activation="relu"))
        if activation == "Dice":
            self.fc.add(Dice()) 
        elif activation == "dice":
            self.fc.add(dice(80))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(2, activation=None))

    def get_emb_din(self, user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list):
        user_profile_feature_embedding = dict()
        for feature in user_profile_list:
            data = user_profile_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            user_profile_feature_embedding[feature] = embedding_layer(data)
        
        target_item_feature_embedding = dict()
        for feature in user_behavior_list:
            data = target_item_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            target_item_feature_embedding[feature] = embedding_layer(data)
        
        hist_behavior_embedding = dict()
        for feature in user_behavior_list:
            data = hist_behavior_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            hist_behavior_embedding[feature] = embedding_layer(data)

        return utils.concat_features(user_profile_feature_embedding), utils.concat_features(target_item_feature_embedding), utils.concat_features(hist_behavior_embedding)
    
    def call(self, user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list, length):
        #Embedding Layer
        user_profile_embedding, target_item_embedding, hist_behavior_emebedding = self.get_emb_din(user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list)
        hist_attn_emb = self.hist_at(target_item_embedding, hist_behavior_emebedding, length)
        join_emb = tf.concat([user_profile_embedding, target_item_embedding, hist_attn_emb], -1)
        logit = tf.squeeze(self.fc(join_emb))
        output = tf.keras.activations.softmax(logit)
        return output, logit

if __name__ == "__main__":
    model = DIN(dict(), dict(), list(), list())
