from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import os
from utils import load_data, preprocess_features, preprocess_adj, construct_feed_dict
from models import GCN, GAT

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

dataset_name = "citeseer"
model_name = "GAT"

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', f'{dataset_name}', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

print(f"Train ratio: {np.sum(train_mask) / len(train_mask):.2f}, Val ratio: {np.sum(val_mask) / len(val_mask):.2f}, Test ratio: {np.sum(test_mask) / len(test_mask):.2f}")

total_edges = adj.data.shape[0]
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

save_name = 'nat_' + model_func.__name__ + '_' + FLAGS.dataset
if not os.path.exists(save_name):
    os.makedirs(save_name)

# Define placeholders
tf.compat.v1.disable_v2_behavior()
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

# Disable eager execution to use placeholders
tf.compat.v1.disable_eager_execution()

# Create model
# for non sparse
model = model_func(placeholders, input_dim=features.shape[1], attack=None, logging=False)

# Initialize session
sess = tf.compat.v1.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders, train=True)
    feed_dict_val.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = sess.run([model.attack_loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

# Init variables
sess.run(tf.compat.v1.global_variables_initializer())

cost_val = []
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders, train=True)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration, _ = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing before attack
test_cost, test_acc, test_duration, save_label = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

save_label = np.argmax(save_label, 1)
tmp = np.zeros_like(y_train)
tmp[np.arange(len(save_label)), save_label] = 1
tmp = y_train + tmp * (1 - np.expand_dims(train_mask, 1))
np.save('label_' + FLAGS.dataset + '.npy', tmp)
print('predicted label saved at ' + 'label_' + FLAGS.dataset + '.npy')

# Save the trained model
saver = tf.compat.v1.train.Saver()  # Use TensorFlow Saver to manage model checkpoints
saver.save(sess, save_name + '/model_trained.ckpt')
print("Trained model saved.")

# Demo code to load and evaluate the saved model
def demo_evaluate():
    with tf.compat.v1.Session() as sess:
        # Restore the trained model
        saver.restore(sess, save_name + '/model_trained.ckpt')
        print("Model restored from checkpoint.")

        # Evaluate on test set
        test_cost, test_acc, test_duration, _ = evaluate(features, support, y_test, test_mask, placeholders)
        print("Demo Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# Run demo
demo_evaluate()