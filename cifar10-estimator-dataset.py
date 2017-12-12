'''
Using tf.estimator and tf.data to train a cnn model in TensorFlow 1.4.

GitHub: https://github.com/secsilm/understaing-datasets-estimators-tfrecords
Chinese blog: 
'''
import tensorflow as tf
import os
import json

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
flags.DEFINE_string('train_dataset', 'train.tfrecords',
                    'Filename of training dataset')
flags.DEFINE_string('eval_dataset', 'eval.tfrecords',
                    'Filename of evaluation dataset')
flags.DEFINE_string('test_dataset', 'test.tfrecords',
                    'Filename of testing dataset')
flags.DEFINE_string('model_dir', 'models/cifar10_cnn_model',
                    'Filename of testing dataset')
FLAGS = flags.FLAGS


def cifar_model_fn(features, labels, mode):
    """Model function for cifar10"""
    # Input layer
    x = tf.reshape(features, [-1, 32, 32, 3])

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[
                             3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV1')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN1')

    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[
                            3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV2')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN2')
    
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[
                            3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV3')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN3')
    
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[
                             3, 3], padding='same', activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV4')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN4')
    
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[
                                    3, 3], strides=2, padding='same', name='POOL1')

    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=3, padding='same',
                             activation=tf.nn.relu, kernel_regularizer=regularizer, name='CONV5')
    x = tf.layers.batch_normalization(
        inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN5')
    
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[
                                    3, 3], strides=2, padding='same', name='POOL2')
    # Dense layer
    x = tf.reshape(x, [-1, 8 * 8 * 128])

    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='DENSE1')
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='DENSE2')
    x = tf.layers.dropout(inputs=x, rate=FLAGS.dropout_rate,
                                training=mode == tf.estimator.ModeKeys.TRAIN, name='DROPOUT')
    logits = tf.layers.dense(inputs=x, units=10,
                             kernel_regularizer=regularizer, name='FINAL')

    # Predicition
    predictions = {
        'classes': tf.argmax(input=logits, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss for train and eval
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    # print('onehot_labels', onehot_labels.shape)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, scope='LOSS')
    # print(labels.shape, predictions['classes'].shape)
    
    accuracy, update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions['classes'], name='accuracy')
    batch_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(labels, tf.int64), predictions['classes']), tf.float32))
    tf.summary.scalar('batch_acc', batch_acc)
    tf.summary.scalar('streaming_acc', update_op)
    # tf.summary.scalar('accuracy', accuracy)
    # eval_metric_ops = {
    #     'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'], name='accuracy')
    # }

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        # tensors_to_log = {
        #     'Accuracy': accuracy,
        #     'My accuracy': my_acc}
        # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': (accuracy, update_op)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def parser(record):
    keys_to_features = {
        'image_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


def save_hp_to_json():
    '''Save hyperparameters to a json file'''
    filename = os.path.join(FLAGS.model_dir, 'hparams.json')
    hparams = FLAGS.flag_values_dict()
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)


def main(unused_argv):

    def train_input_fn():
        train_dataset = tf.data.TFRecordDataset(FLAGS.train_dataset)
        train_dataset = train_dataset.map(parser)
        train_dataset = train_dataset.repeat(FLAGS.num_epochs)
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()

        features, labels = train_iterator.get_next()
        return features, labels

    def eval_input_fn():
        eval_dataset = tf.data.TFRecordDataset(FLAGS.eval_dataset)
        eval_dataset = eval_dataset.map(parser)
        # eval_dataset = eval_dataset.repeat(FLAGS.num_epochs)
        eval_dataset = eval_dataset.batch(FLAGS.batch_size)
        eval_iterator = eval_dataset.make_one_shot_iterator()
        features, labels = eval_iterator.get_next()
        return features, labels

    cifar10_classifier = tf.estimator.Estimator(
        model_fn=cifar_model_fn, model_dir=FLAGS.model_dir)

    # Train
    cifar10_classifier.train(input_fn=train_input_fn)

    # Evaluation
    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    tf.logging.info('Saving hyperparameters ...')
    save_hp_to_json()


if __name__ == '__main__':
    tf.app.run()
