# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:03:41 2018

@author: maliang
"""
from math import cos, pi

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot as plt


def true_distribution(t, freq=10):
    return cos(2 * pi * freq * t)


def sample_from_true_distribution(sample_points):
    sample_signal = list()
    for value in sample_points:
        sample_signal.append(true_distribution(value))
    return np.array(sample_signal)


def generate_true_samples(n_true_samples=10000,
                          sample_length=500,
                          true_sample_range=1.0):
    x_true = np.zeros((1, sample_length))
    for i in range(n_true_samples):
        sample_start = np.random.rand(1)
        x = np.linspace(sample_start, sample_start + true_sample_range,
                        sample_length)
        y = sample_from_true_distribution(x).reshape(1, -1)
        x_true = np.r_[x_true, y]
    x_true = x_true[1:, :]
    return x_true


def plot_true_sample(n_samples_plot, true_samples):
    for i in range(n_samples_plot):
        plt.plot(true_samples[i, :], label='# ' + str(i + 1))
    plt.legend()
    plt.show()


##test code
#true_samples = generate_true_samples(10)
#plot_true_sample(5, true_samples)


def Generator_model():
    model = Sequential()
    model.add(Dense(300, input_dim=100, activation='tanh'))
    model.add(Dense(500, activation='tanh'))
    return model


def Discriminator_model():
    model = Sequential()
    model.add(Dense(300, input_dim=500, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def Generator_containing_Discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def train(BATCH_SIZE, n_epochs):
    d_losses = list()
    g_losses = list()

    x_train = generate_true_samples(10000)

    D = Discriminator_model()
    G = Generator_model()
    D_on_G = Generator_containing_Discriminator(G, D)

    D_optimizer = SGD(lr=0.01, nesterov=True)
    G_optimizer = SGD(lr=0.01, nesterov=True)

    G.compile(loss='binary_crossentropy', optimizer='SGD')
    D_on_G.compile(loss='binary_crossentropy', optimizer=G_optimizer)

    D.trainable = True
    D.compile(loss='binary_crossentropy', optimizer=D_optimizer)

    for epoch in range(n_epochs):
        print('Epoch', str(epoch + 1))

        n_batches = int(x_train.shape[0] / BATCH_SIZE)
        print('# of batches', str(n_batches))

        for i_batch in range(n_batches):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            true_sample_batch = x_train[i_batch * BATCH_SIZE:(
                i_batch + 1) * BATCH_SIZE, :]

            generated_signals = G.predict(noise, verbose=0)

            X = np.concatenate((true_sample_batch, generated_signals))

            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            d_loss = D.train_on_batch(X, y)
            d_losses.append(d_loss)
            print('batch %d d_loss : %f' % (i_batch, d_loss))

            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            D.trainable = False
            g_loss = D_on_G.train_on_batch(noise, [1] * BATCH_SIZE)
            g_losses.append(g_loss)

            D.trainable = True
            print('batch %d g_loss : %f' % (i_batch, g_loss))

        # 每个epoch结束生成一次信号并保存
        one_noise = np.random.uniform(-1, 1, size=(1, 100))
        one_generated_signal = G.predict(one_noise, verbose=0).reshape(500, )
        plt.plot(one_generated_signal, lw=0.6)
        plt.ylim(-1.5, 1.5)
        plt.axhline(-1, linestyle='--', lw=0.6, color='k')
        plt.axhline(1, linestyle='--', lw=0.6, color='k')
        plt.savefig('GAN/epoch_' + str(epoch + 1) + '.jpg', dpi=60)
        plt.clf()
        plt.close()
    return g_losses, d_losses


g_losses, d_losses = train(BATCH_SIZE=100, n_epochs=50)
plt.plot(g_losses, label='g_loss')
plt.plot(d_losses, label='d_loss')
plt.legend()
plt.savefig('GAN/loss.jpg', dpi=300)
plt.clf()
plt.close()
