# import the necessary packages

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from data import polyvore_dataset, DataGenerator
from utils import Config

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

# from modeldense import dense_121

    # Use GPU

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        nnet_inputs=keras.layers.Input(shape=(224,224,3),name='input_picture')
        z=keras.layers.Conv2D(64, 7, strides=2, padding='same',use_bias=False)(nnet_inputs)
        z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
        z=keras.layers.Activation('relu')(z)
        z=keras.layers.MaxPooling2D(3,padding='same',strides=2)(z)

        node_number=32
        # dense block 1
        for i in range (6):
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(node_number*4,1, strides=1, padding='same',use_bias=False)(z)
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(32,3, strides=1, padding='same',use_bias=False)(z)
            node_number=node_number+8
            # Trans Layer
            # compression rate=0.8
        z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
        z=keras.layers.Activation('relu')(z)
        z=keras.layers.Conv2D(int(node_number*0.8),1, strides=1, padding='same',use_bias=False)(z)
        z=keras.layers.AveragePooling2D(2,padding='same',strides=2)(z)
        node_number=int(node_number*0.8)

            # dense block 2
        for i in range (6):
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(node_number*4,1, strides=1, padding='same',use_bias=True)(z)
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(32,3, strides=1, padding='same',use_bias=True)(z)
            node_number=node_number+8

            # Trans Layer
            # compression rate=0.8
        z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
        z=keras.layers.Activation('relu')(z)
        z=keras.layers.Conv2D(int(node_number*0.8),1, strides=1, padding='same',use_bias=True)(z)
        z=keras.layers.AveragePooling2D(2,padding='same',strides=2)(z)
        node_number=int(node_number*0.8)

        for i in range (12):
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(node_number*4,1, strides=1, padding='same',use_bias=True)(z)
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(32,3, strides=1, padding='same',use_bias=True)(z)
            node_number=node_number+8

            # compression rate=0.8
        z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
        z=keras.layers.Activation('relu')(z)
        z=keras.layers.Conv2D(int(node_number*0.8),1, strides=1, padding='same',use_bias=True)(z)
        z=keras.layers.AveragePooling2D(2,padding='same',strides=2)(z)
        node_number=int(node_number*0.8)

        for i in range (8):
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(node_number*4,1, strides=1, padding='same',use_bias=True)(z)
            z=keras.layers.BatchNormalization(axis=1,epsilon=0.001)(z)
            z=keras.layers.Activation('relu')(z)
            z=keras.layers.Conv2D(32,3, strides=1, padding='same',use_bias=True)(z)
            node_number=node_number+8

        z = keras.layers.GlobalAveragePooling2D()(z)
        z = keras.layers.Dense(n_classes , activation='softmax')(z)

        model=keras.Model(inputs=nnet_inputs,outputs=z)


        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model,to_file='kiya.png',show_shapes=True,show_layer_names=True)
# result=model.fit(data_new,data_label,epochs=40,batch_size=16,validation_split=0.2)
    # training
    result=model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,epochs=Config['num_epochs'],
                        workers=Config['num_workers']
                        )



    loss=result.history['loss']
    val_loss=result.history['val_loss']
    import numpy as np
    print(np.shape(loss))
    epochs=np.arange(len(loss))
    plt.figure()
    plt.plot(epochs,loss,label='loss')
    plt.plot(epochs,val_loss,label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_dense.png',dpi=256)

    acc=result.history['accuracy']
    val_acc=result.history['val_accuracy']
    print(acc)
    print(val_acc)

    plt.figure()
    plt.plot(epochs,acc,label='acc')
    plt.plot(epochs,val_acc,label='val_acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('percentage')
    plt.savefig('accuracy_dense.jpg',dpi=256)