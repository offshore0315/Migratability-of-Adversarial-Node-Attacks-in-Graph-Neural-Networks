from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
import copy
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

dataset_name = "cora"
model_name = "GAT"

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', f'nat_{model_name}_{dataset_name}', 'saved model directory')
flags.DEFINE_string('dataset', f'{dataset_name}', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_integer('steps', 20, 'Number of steps to attack')  # attack epoch
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of steps).')
flags.DEFINE_string('method', 'PGD', 'attack method, PGD or CW')
flags.DEFINE_float('perturb_ratio', 0.05, 'perturb ratio of total edges.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
total_edges = adj.data.shape[0] / 2
n_node = adj.shape[0]

# Some preprocessing
features = preprocess_features(features)
# for non sparse
features = sp.coo_matrix((features[1], (features[0][:, 0], features[0][:, 1])), shape=features[2]).toarray()

support = preprocess_adj(adj)
# for non sparse
support = [sp.coo_matrix((support[1], (support[0][:, 0], support[0][:, 1])), shape=support[2]).toarray()]
num_supports = 1
# Dynamically select model based on model_name
if model_name == "GCN":
    model_func = GCN
elif model_name == "GAT":
    model_func = GAT

# Define placeholders
placeholders = {
    'lmd': tf.placeholder(tf.float32),
    'mu': tf.placeholder(tf.float32),
    's': [tf.placeholder(tf.float32, shape=(n_node, n_node)) for _ in range(num_supports)],
    'adj': [tf.placeholder(tf.float32, shape=(n_node, n_node)) for _ in range(num_supports)],
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'label_mask_expand': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
# for non sparse
model = model_func(placeholders, input_dim=features.shape[1], attack=FLAGS.method, logging=False)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = sess.run([model.attack_loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

# Load model
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=''))
checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
if checkpoint:
    try:
        saver.restore(sess, checkpoint)
    except tf.errors.NotFoundError as e:
        print("Checkpoint restoration failed. Please ensure that the model structure matches the checkpoint.")
        print(e)
else:
    raise FileNotFoundError("Checkpoint not found: {}".format(FLAGS.model_dir))

adj = adj.toarray()
lmd = 1
eps = total_edges * FLAGS.perturb_ratio
xi = 1e-5

## results before attack
test_cost, test_acc, test_duration = evaluate(features, support, y_train, train_mask, placeholders)
print("Train set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, support, y_val, val_mask, placeholders)
print("Validation set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
before_attack_result_str = f"Test set results: cost={test_cost:.5f}, accuracy={test_acc:.5f}, time={test_duration:.5f}"

label = y_train
label_mask = train_mask + test_mask

original_support = copy.deepcopy(support)
feed_dict = construct_feed_dict(features, support, label, label_mask, placeholders)
feed_dict.update({placeholders['lmd']: lmd})
feed_dict.update({placeholders['dropout']: FLAGS.dropout})
feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
# feed_dict.update({placeholders['s'][i]: np.random.uniform(size=(n_node,n_node)) for i in range(num_supports)})
feed_dict.update({placeholders['s'][i]: np.zeros([n_node, n_node]) for i in range(num_supports)})

if FLAGS.method == 'CW':
    label_mask_expand = np.tile(label_mask, [label.shape[1], 1]).transpose()
    feed_dict.update({placeholders['label_mask_expand']: label_mask_expand})
    C = 0.1
else:
    C = 200  # initial learning rate

if os.path.exists('label_' + FLAGS.dataset + '.npy'):
    label = np.load('label_' + FLAGS.dataset + '.npy')
else:
    ret = sess.run(model.outputs, feed_dict=feed_dict)
    ret = np.argmax(ret, 1)
    label = np.zeros_like(label)
    label[np.arange(label.shape[0]), ret] = 1
    np.save('label_' + FLAGS.dataset + '.npy', label)
feed_dict.update({placeholders['labels']: label})

# Demo function to evaluate restored model
def demo_evaluate():
    with tf.Session() as sess:
        try:
            # Restore the trained model
            saver.restore(sess, checkpoint)
            print("Model restored from checkpoint.")
        except tf.errors.NotFoundError as e:
            print("Checkpoint restoration failed. Please ensure that the model structure matches the checkpoint.")
            print(e)
            return

        # Evaluate on test set
        test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
        print("Demo Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


print("Running demo_evaluate")
demo_evaluate()

print('{} attack begin:'.format(FLAGS.method))
for epoch in range(FLAGS.steps):

    t = time.time()
    # mu = C/np.sqrt(np.sqrt(epoch+1))
    mu = C / np.sqrt(epoch + 1)
    feed_dict.update({placeholders['mu']: mu})

    # s \in [0,1]
    if FLAGS.method == 'CW':
        a, support, l, g = sess.run([model.a, model.placeholders['support'], model.loss, model.Sgrad],
                                    feed_dict=feed_dict)
        # print('loss:', l)
    elif FLAGS.method == 'PGD':
        a, support, S, g = sess.run([model.a, model.placeholders['support'], model.upper_S_real, model.Sgrad],
                                    feed_dict=feed_dict)
    else:
        raise ValueError('invalid attack method: {}'.format(FLAGS.method))
    upper_S_update = bisection(a, eps, xi)

    feed_dict.update({placeholders['s'][i]: upper_S_update[i] for i in range(num_supports)})

    upper_S_update_tmp = upper_S_update[:]
    if epoch == FLAGS.steps - 1:
        acc_record, support_record, p_ratio_record = [], [], []
        for i in range(10):
            print('random start!')
            randm = np.random.uniform(size=(n_node, n_node))
            upper_S_update = np.where(upper_S_update_tmp > randm, 1, 0)
            feed_dict.update({placeholders['s'][i]: upper_S_update[i] for i in range(num_supports)})
            support = sess.run(model.placeholders['support'], feed_dict=feed_dict)
            cost, acc, duration = evaluate(features, support, y_test, test_mask, placeholders)
            pr = np.count_nonzero(upper_S_update[0]) / total_edges
            if pr <= FLAGS.perturb_ratio:
                acc_record.append(acc)
                support_record.append(support)
                p_ratio_record.append(pr)
            print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(cost),
                  "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
            print("perturb ratio", pr)
            print('random end!')
        # Validation
        support = support_record[np.argmin(np.array(acc_record))]

    cost, acc, duration = evaluate(features, support, y_test, test_mask, placeholders)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(cost),
          "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

print("attack Finished!")
print("perturb ratio", np.count_nonzero(upper_S_update[0]) / total_edges)

# Save adversarial samples
adv_sample_path = 'adversarial_support_{}_{}.npy'.format(FLAGS.model_dir.split('_')[1], FLAGS.dataset)
np.save(adv_sample_path, support)
print('Adversarial samples saved at ' + adv_sample_path)

# Testing after attack
test_cost, test_acc, test_duration = evaluate(features, support, y_train, train_mask, placeholders)
print("Train set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, support, y_val, val_mask, placeholders)
print("Validation set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

def demo2_attack():
    with tf.Session() as sess:
        try:
            # Restore the trained model
            saver.restore(sess, checkpoint)
            print("Model restored from checkpoint.")
        except tf.errors.NotFoundError as e:
            print("Checkpoint restoration failed. Please ensure that the model structure matches the checkpoint.")
            print(e)
            return

        # Load adversarial samples
        adv_sample_path = 'adversarial_support_{}_{}.npy'.format(FLAGS.model_dir.split('_')[1], FLAGS.dataset)
        if os.path.exists(adv_sample_path):
            adversarial_support = np.load(adv_sample_path, allow_pickle=True)
            print("Adversarial samples loaded from", adv_sample_path)
        else:
            print("Adversarial sample file not found at", adv_sample_path)
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
        print("After attack - Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print("Running demo2_attack function")
demo2_attack()
print(before_attack_result_str)