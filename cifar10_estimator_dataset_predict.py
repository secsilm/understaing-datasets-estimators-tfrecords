'''
Using saved model to predict new image with estimator.predict().
'''
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from cifar10_estimator_dataset import get_estimator

flags = tf.app.flags
flags.DEFINE_string('saved_model_dir', 'models/adam',
                    'Output directory for model and training stats.')
FLAGS = flags.FLAGS


def infer(argv=None):
    '''Run the inference and return the result.'''
    config = tf.estimator.RunConfig()
    config = config.replace(model_dir=FLAGS.saved_model_dir)
    estimator = get_estimator(config)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=load_image(), shuffle=False)
    result = estimator.predict(input_fn=predict_input_fn)
    for r in result:
        print(r)
    # return result


# def predict_input_fn():
#     '''Input function for prediction.'''
#     with tf.variable_scope('TEST_INPUT'):
#         image = tf.constant(load_image(), dtype=tf.float32)
#         dataset = tf.data.Dataset.from_tensor_slices((image,))
#         return dataset.batch(1).make_one_shot_iterator().get_next()


def load_image():
    '''Load image into numpy array.'''
    images = np.zeros((10, 3072), dtype='float32')
    for i, file in enumerate(Path('predict-images/').glob('*.png')):
        image = np.array(Image.open(file)).reshape(3072)
        images[i, :] = image
    return images


if __name__ == '__main__':
    tf.app.run(main=infer)
