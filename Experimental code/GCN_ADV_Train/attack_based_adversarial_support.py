from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
import numpy as np
import os

from utils import *
from models import GCN, GAT

# Disable eager execution for TensorFlow 2 compatibility
tf.disable_eager_execution()

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

migrate_dataset_name = "citeseer"  # 源，指对抗网络
dataset_name = "citeseer"  # 目标，指保存的模型
migrate_model_name = "GAT" # 源，指对抗网络
model_name = "GCN"  # 目标，指保存的模型

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', f'nat_{model_name}_{dataset_name}', 'saved model directory')
flags.DEFINE_string('dataset', f'{dataset_name}', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('adv_sample_path', f'adversarial_support_{migrate_model_name}_{migrate_dataset_name}_to_{dataset_name}.npy', 'Path to the adversarial samples file')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of steps).')
flags.DEFINE_string('method', 'PGD', 'attack method, PGD or CW')
flags.DEFINE_float('perturb_ratio', 0.05, 'perturb ratio of total edges.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
n_node = adj.shape[0]

# Some preprocessing
features = preprocess_features(features)
# For non-sparse representation
features = sp.coo_matrix((features[1], (features[0][:, 0], features[0][:, 1])), shape=features[2]).toarray()

support = preprocess_adj(adj)
# For non-sparse representation
support = [sp.coo_matrix((support[1], (support[0][:, 0], support[0][:, 1])), shape=support[2]).toarray()]
num_supports = 1
# Dynamically select model based on model_name
if model_name == "GCN":
    model_func = GCN
elif model_name == "GAT":
    model_func = GAT

# Define placeholders
placeholders = {
    's': [tf.compat.v1.placeholder(tf.float32, shape=(n_node, n_node)) for _ in range(num_supports)],
    'adj': [tf.compat.v1.placeholder(tf.float32, shape=(n_node, n_node)) for _ in range(num_supports)],
    'support': [tf.compat.v1.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.compat.v1.placeholder(tf.float32, shape=features.shape),
    'lmd': tf.compat.v1.placeholder(tf.float32),
    'mu': tf.compat.v1.placeholder(tf.float32),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features.shape[1], logging=False)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Load model
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=''))
checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
if checkpoint:
    try:
        saver.restore(sess, checkpoint)
        print("Model restored from checkpoint at '{}'.".format(checkpoint))
    except tf.errors.NotFoundError as e:
        print("Checkpoint restoration failed. Please ensure that the model structure matches the checkpoint.")
        print(e)
        exit()
else:
    raise FileNotFoundError("Checkpoint not found in directory: {}".format(FLAGS.model_dir))

# Evaluate on test set before attack
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Before attack - Test set results:",
          "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "time=", "{:.5f}".format(test_duration))

def attack_with_adversarial_samples():
    # Load adversarial samples
    adv_sample_path = FLAGS.adv_sample_path
    if os.path.exists(adv_sample_path):
        adversarial_support = np.load(adv_sample_path, allow_pickle=True)
        print("Adversarial samples loaded from '{}'.".format(adv_sample_path))
    else:
        print("Adversarial sample file not found at '{}'.".format(adv_sample_path))
        return

    # Ensure adversarial samples are applied correctly
    if isinstance(adversarial_support, np.ndarray) and adversarial_support.ndim == 3:
        adversarial_support = [adversarial_support[i] for i in range(adversarial_support.shape[0])]
    elif isinstance(adversarial_support, np.ndarray) and adversarial_support.ndim == 2:
        adversarial_support = [adversarial_support]
    else:
        print("Unexpected format for adversarial samples.")
        return

    # Evaluate on test set after attack
    test_cost, test_acc, test_duration = evaluate(features, adversarial_support, y_test, test_mask, placeholders)
    print("After attack - Test set results:",
          "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "time=", "{:.5f}".format(test_duration))

print("Running attack_with_adversarial_samples function")
attack_with_adversarial_samples()
