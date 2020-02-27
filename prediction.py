logits = alexnet(X, weights, biases)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    acc = []
    sess.run(init)
    for i in range(100, 29001, 100):
        acc.append(sess.run(accuracy, feed_dict = { X : x_train[i - 100 : i], Y : y_train_encoded[i - 100 : i] }))
print('Accuracy on Training Data: ' + str(sum(acc) * 100 / len(acc)) + '%')

with tf.Session() as sess:
    acc = []
    sess.run(init)
    for i in range(100, 29001, 100):
        acc.append(sess.run(accuracy, feed_dict = { X : x_train[i - 100 : i], Y : y_train_encoded[i - 100 : i] }))
print('Accuracy on Training Data: ' + str(sum(acc) * 100 / len(acc)) + '%')

y_pred = [y_test[list(i).index(max(list(i)))] for i in y_test]
display_images(next_batch(12, x_test, y_pred), 'Predictions')
