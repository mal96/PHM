# 描述一层网络的方式: tuple -- LSTM层: ('L', n_units, depth)
#                           Dense层:('D', n_units, depth)
#                           终止层:  ('T', depth)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import mean_absolute_error
import pandas as pd


class neural_network(object):
    def __init__(self, net_structure, input_shape):
        self.model = Sequential()
        for layer_paras in net_structure:
            layer_name = layer_paras[0]
            # LSTM层
            if layer_name == 'L':
                layer_n_units = layer_paras[1]
                layer_depth = layer_paras[2]
                if layer_depth == 0:
                    self.model.add(
                        LSTM(
                            units=layer_n_units,
                            input_shape=input_shape,
                            return_sequences=True))
                elif layer_depth == 1:
                    self.model.add(
                        LSTM(units=layer_n_units, return_sequences=True))
                elif layer_depth == 2:
                    self.model.add(
                        LSTM(units=layer_n_units, return_sequences=False))
                else:
                    raise Exception('LSTM depth overflow')
            # Dense层
            elif layer_name == 'D':
                layer_n_units = layer_paras[1]
                self.model.add(
                    Dense(units=layer_n_units, activation='relu'))
            # 终止层
            elif layer_name == 'T':
                self.model.add(Dense(1, activation=None))
            else:
                raise Exception('Layer-name Error')
    
    # 模型编译
    def compile_model(self):
        self.model.compile(optimizer='adam', loss=mean_absolute_error)

    # 模型训练 获得损失变化
    def fit_model(self, X, y, batch_size, nb_epoch, val_X, val_y):
        hist = self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=nb_epoch,
            validation_data=(val_X, val_y),
            verbose=0)
        self.loss = hist.history['loss']
        self.val_loss = hist.history['val_loss']

    # 模型性能评估 并输出reward
    def evaluate_model(self, scale=10.0):
        self.val_accuracy = scale / min(self.val_loss)
        return self.val_accuracy
    
    # 保存模型及训练信息
    def save_model(self, save_path):
        self.model.save(save_path)
