import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import argparse
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from keras_retinanet.preprocessing.csv_generator import CSVGenerator

import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def parse_args():
    parser = argparse.ArgumentParser(description='testing script.')
    parser.add_argument('--annotations', default='train/annotations.csv', help='Path to annotations')
    parser.add_argument('--classes', default='train/meditatio_classes.csv', help='Path to a CSV file containing class label mapping (required)')
    parser.add_argument('--val_path', default='test/images/Med_Ms_p0132_M_5_rgb_extracted.jpg', help='Path to image')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).',
                        default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')

    return parser.parse_args()


def getCoord(b, size, scale):
    mylist=[]
    for coord in b:
        if coord>size:
            mylist.append(coord)
    if len(mylist):
        scaled = [int(x) for x in b / scale]
        return np.asarray(scaled)
    else:
        return b


def detectAlphabets(imageToRecognize):
    args = parse_args()
    args.val_path = imageToRecognize
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    keras.backend.tensorflow_backend.set_session(get_session())
    model = keras.models.load_model('../snapshots/resnet50_csv_wtext.h5', custom_objects=custom_objects)
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()
    #
    # # create a generator for testing data
    test_generator = CSVGenerator(
        csv_data_file=args.annotations,
        csv_class_file=args.classes,
        image_data_generator=test_image_data_generator,
        batch_size=args.batch_size
    )
    # index = 0
    # load image
    image = read_image_bgr(args.val_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    print('detections:', detections)
    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
    print("label=", predicted_labels)
    # correct for image scale
    scaled_detection = detections[0, :, :4] / scale

    # visualize detections
    recognized = {}

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.4:
            continue
        b = scaled_detection[idx, :4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)

        caption = test_generator.label_to_name(label)
        cv2.putText(draw, caption, (b[0], b[1] - 1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        recognized[caption] = b
        print(caption + ":" + str(score))

    plt.imshow(draw)
    plt.show()
    return recognized, draw
if __name__ == '__main__':
    # parse arguments

    imageToRecognize = os.path.join('test/images', 'Med_MS_p0165_F_1_rgb_extracted.jpg')
    recognized, image = detectAlphabets(imageToRecognize)




