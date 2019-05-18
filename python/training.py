import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import pickle
import argparse


def getEncoder(images, alpha, xavier_wts, inputs_, is_training = True, reuse=False):
    ### Encoder  
    with tf.variable_scope("encoder",reuse=reuse):

        # input: 224 * 224 * num_channels    
        x = tf.layers.conv2d(inputs=inputs_, filters=8, kernel_size=5, strides=2, activation=None, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.maximum(alpha * x, x)

        #print(x.shape.as_list())

        # input: 112 * 112 * _   
        x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=5, strides=2, activation=None, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.maximum(alpha * x, x)
        #print(x.shape.as_list())

        # input: 56 * 56 * _
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, strides=2, activation=None, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.maximum(alpha * x, x)
        #print(x.shape.as_list())

        # input: 28 * 28 * _
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=5, strides=2, activation=None, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.maximum(alpha * x, x)
        #print(x.shape.as_list())

        # input: 14 * 14 * _
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=5, strides=2, activation=None, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x = tf.layers.batch_normalization(inputs=x, training=is_training)
        x = tf.maximum(alpha * x, x)
        #print(x.shape.as_list())

        # Output: 7 * 7 * _
        return x

def getDecoder(encoder_input, alpha, xavier_wts, is_training=True, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        
        # input: 7 * 7 * _
        x1 = tf.image.resize_nearest_neighbor(encoder_input, (14, 14))

        x2 = tf.layers.conv2d(inputs=x1, filters=64, activation=None, kernel_size=5, strides=1, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_training)
        x2 = tf.maximum(alpha * x2, x2)

        # input: 14 * 14 * _ 
        x3 = tf.image.resize_nearest_neighbor(x2, (28, 28))
        x3 = tf.layers.conv2d(inputs=x3, filters=32, activation=None, kernel_size=5, strides=1, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_training)
        x3 = tf.maximum(alpha * x3, x3)        

        # input: 28 * 28 * _ 
        x4 = tf.image.resize_nearest_neighbor(x3, (56, 56))
        x4 = tf.layers.conv2d(inputs=x4, filters=16, activation=None, kernel_size=5, strides=1, padding='same', kernel_initializer=xavier_wts, use_bias=False)
        x4 = tf.layers.batch_normalization(inputs=x4, training=is_training)
        x4 = tf.maximum(alpha * x4, x4)        

        # input: 56 * 56 * _ 
        x5 = tf.image.resize_nearest_neighbor(x4, (112, 112))
        x5 = tf.layers.conv2d(inputs=x5, filters=8, activation=None, kernel_size=5, strides=1, padding='same', kernel_initializer=xavier_wts, use_bias=False )
        x5 = tf.layers.batch_normalization(inputs=x5, training=is_training)
        x5 = tf.maximum(alpha * x5, x5)        

        # input: 112 * 112 * _
        x5 = tf.image.resize_nearest_neighbor(x4, (224, 224))
        x5 = tf.layers.conv2d(inputs=x5, filters=1, activation=None, kernel_size=5, strides=1, padding='same', kernel_initializer=xavier_wts, use_bias=False)

        return x5

def image_generator(images, batch_size):
    start_index = 0    
    while start_index < len(images):
        end_index = start_index + batch_size
        yield images[start_index:end_index,:,:,:]
        start_index = end_index


# In[ ]:
def main(args):
    images_list = []
    failed_list = []

    for name in os.listdir(args.input_path):
        if name.endswith(".jpg"):
            filename = os.path.join(args.input_path, name)
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            #lastSlash = filename.rfind('/')
            #name = filename[lastSlash+1:]
            try:
                image = cv2.resize(image, (224, 224))
                images_list.append(image)
            except:
                print("could not process image: "+name)
                failed_list.append(name)

    train_images = np.array(images_list)

    if len(train_images) == 0:
        raise Exception("No input images found!")

    mean_imgs = np.mean(train_images)
    maxval = np.max(train_images)

    train_rescaled_images = (train_images - mean_imgs)/maxval
    train_rescaled_images_arr = np.reshape(train_rescaled_images, newshape=(-1,224,224,1))

    inputs_ = tf.placeholder(tf.float32, (None, 224, 224, 1), name='inputs')

    targets_ = tf.placeholder(tf.float32, (None, 224, 224, 1), name='targets')
    lr_ = tf.placeholder(tf.float32)

    xavier_wts = tf.contrib.layers.xavier_initializer()

    enc_output = getEncoder(inputs_, args.alpha, xavier_wts, inputs_)
    dec_output = getDecoder(enc_output, args.alpha, xavier_wts)

    loss = tf.reduce_mean(tf.squared_difference(targets_, dec_output))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = tf.train.AdamOptimizer(lr_).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    start_epochs = 0
    saver = tf.train.Saver()

    foo = list(image_generator(train_rescaled_images_arr, batch_size=args.batch))
    for epoch_n in range(start_epochs, args.end_epochs+1):
        print("starting epoch: "+str(epoch_n+1))
        counter = start_epochs

        for img in tqdm(foo):
            _ , train_loss = sess.run([opt, loss], feed_dict={inputs_:img, targets_:img, lr_:args.learn_rate})

        print("  train_loss: "+str(train_loss))

        if(epoch_n % args.save_every_epochs == 0):
            print("saving model at epochs: ", epoch_n)
            saver.save(sess, args.model_name, global_step=epoch_n)


if __name__ == "__main__":
#     dataInput = 'C:\\Users\\Jelly\\dlcollabrate-master\\bag_train\\*.jpg'
# alpha = 0.5
# BATCH = 128
# end_epochs = 200
# learn_rate = 0.0002
# save_every_epochs = 50
# model_name = "./saved_models/grayscale_model"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_path', required=True, help='Input files')
    parser.add_argument('-a', dest='alpha', default=0.5, help='Alpha weight')
    parser.add_argument('-b', dest='batch', default=128, help='')
    parser.add_argument('-e', dest='end_epochs', default=200, help='Name of the output COS bucket')
    parser.add_argument('-l', dest='learn_rate', default=0.0002, help='Name of the output COS bucket')
    parser.add_argument('-s', dest='save_every_epochs', default=50, help='Name of the output COS bucket')
    parser.add_argument('-o', dest='model_name', required=True, help='Name of the output COS bucket')
    args = parser.parse_args()

    main(args)

