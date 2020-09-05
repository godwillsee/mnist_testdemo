#import input_data
import tensorflow as tf
from tensorflow.keras import datasets
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_test, y_test) = datasets.mnist.load_data()

def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000).batch(128)
train_db = train_db.map(process)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
test_db = test_db.map(process)

w1 = tf.Variable(tf.random.truncated_normal([784,256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256.128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

for each in range(100):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            y_onehot = tf.one_hot(y, depth=10)

            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
        w2.assign_sub(lr*grads[2])
        b2.assign_sub(lr*grads[3])
        w3.assign_sub(lr*grads[4])
        b3.assign_sub(lr*grads[5])

        if step % 100 == 0:
            print(each,step, 'loss:', float(loss))

    total_correct, total_num = 0, 0
    for step, (x, y) in enumerate(test_db):

        x = tf.reshape(x, [-1, 28*28])

        h1 = tf.nn.relu(x@w1 + b1)
        h2 = tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3

        prob = tf.nn.softmax(out, axis=1)

        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct / total_num
    print('test acc', acc)