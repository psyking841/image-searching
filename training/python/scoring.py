# coding: utf-8

import numpy as np
import tensorflow as tf
import glob
import cv2
from tqdm import tqdm
import pandas as pd
import pickle
from numpy.linalg import norm
import argparse

def image_generator(images, batch_size):
    start_index = 0    
    while start_index < len(images):
        end_index = start_index + batch_size
        yield images[start_index:end_index,:,:,:]
        start_index = end_index

def main(args):
    images_list = []
    labels_list = []
    failed_list = []

    for filename in glob.glob(args.scoreInput):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        lastSlash = filename.rfind('/')
        name = filename[lastSlash+1:]
        try:
            image = cv2.resize(image, (224, 224))
            images_list.append(image)
            labels_list.append(name)
        except:
            print("could not process image: "+name)
            failed_list.append(name)

    print("saving failed images list...")
    np.save('failed_images_list.dat', failed_list)

    score_images = np.array(images_list)

    mean_imgs = np.mean(score_images)
    maxval = np.max(score_images)

    score_rescaled_images = (score_images - mean_imgs)/maxval
    score_rescaled_images_arr = np.reshape(score_rescaled_images, newshape=(-1,224,224,1)).astype('float32')

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(args.modelName + ".meta")
    new_saver.restore(sess, args.modelName)
    graph = tf.get_default_graph()
    enc_output = graph.get_tensor_by_name("encoder/Maximum_4:0")

    #score_input = tf.placeholder(tf.float32, (None, 224, 224, 1), name="inputs")

    size_deep_conv = 49
    num_deep_filters = 128
    unrolled = tf.reshape(enc_output, [-1, num_deep_filters*size_deep_conv])
    num_score = score_images.shape[0]

    encoder_score_output = np.zeros(shape=(num_score,num_deep_filters*size_deep_conv))

    counter = 0
    for imgs in tqdm(list(image_generator(score_rescaled_images_arr, args.batch))):
        print(imgs.shape)
        print(imgs.dtype)
        batch_out = sess.run(unrolled, feed_dict={"inputs:0" : imgs})
        encoder_score_output[counter: counter + args.batch, :] = batch_out
        counter += args.batch
    print('done..')
    print("encoder score output shape: ",encoder_score_output.shape)

    with open("score_vector_bank.dat","wb") as f:
        pickle.dump(encoder_score_output, f)

    num_images = encoder_score_output.shape[0]
    num_vectors= encoder_score_output.shape[1]
    magnitudes = np.zeros(num_images)
    for i in range(num_images):
        magnitudes[i] = norm(encoder_score_output[i,:])

    matrix = tf.placeholder(shape=(num_images, num_vectors), dtype=tf.float32)
    dotMatrixOp = tf.matmul(matrix, matrix, transpose_a=False, transpose_b=True)

    sess2 = tf.Session()
    sess2.run(tf.global_variables_initializer())

    dotMatrix = sess2.run(dotMatrixOp, feed_dict={matrix: encoder_score_output})

    dotMatrixNormalized = np.zeros((num_images,num_images))

    for i in tqdm(range(num_images)):
        for j in range(num_images):
            dotMatrixNormalized[i,j] = dotMatrix[i,j]/(magnitudes[i] * magnitudes[j])


    similar_image_labels = []
    #show_image_labels = []
    for i, item in enumerate(dotMatrixNormalized):
        most_similar = np.argsort(item)
        top_n_labels = most_similar[::-1][1:args.n_top+1]
        label = []
        for j in top_n_labels:
            label.append(labels_list[j])
        similar_image_labels.append(label)
    #    show_image_labels.append(top_n_labels)

    final_result = pd.DataFrame({"images":labels_list, "similar_images":similar_image_labels})
    final_result.to_csv(args.result_file,index=False)

if __name__ == "__main__":
    # scoreInput = 'C:\\Users\\Jelly\\dlcollabrate-master\\bag_test\\*.jpg'
    # scoreBATCH = 256
    # modelName = "./saved_models/grayscale_model-200"
    # n_top = 10
    # result_file = "final_result.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='scoreInput', required=True, help='Input files')
    parser.add_argument('-b', dest='batch', default=256, help='')
    parser.add_argument('-m', dest='modelName', required=True, help='Name of the final model')
    parser.add_argument('-n', dest='n_top', default=10, help='Number of similar images')
    parser.add_argument('-o', dest='result_file', required=True, help='Name of the result file')
    args = parser.parse_args()

    main(args)




