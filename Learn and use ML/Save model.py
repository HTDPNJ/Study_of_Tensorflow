import tensorflow as tf
from tensorflow import keras
import os
(train_images,train_labels),(test_images,test_labels)=tf.keras.datasets.mnist.load_data()

train_labels=train_labels[:1000]
test_labels=test_labels[:1000]

train_images=train_images[:1000].reshape(-1,28*28)/255.0
test_images=test_images[:1000].reshape(-1,28*28)/255.0

def create_model():
    model=tf.keras.models.Sequential([
        keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model

# model=create_model()
# model.summary()

# checkpoint_path = "cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create checkpoint callback  训练模型并将其传递ModelCheckpoint回调
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model = create_model()
#
# model.fit(train_images, train_labels,  epochs = 10,
#           validation_data = (test_images,test_labels),
#           callbacks = [cp_callback])  # pass callback to training
#
# loss, acc = model.evaluate(test_images, test_labels)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#
#
# model.load_weights(checkpoint_path)  #然后从检查点加载权重，并重新评估：
# loss,acc=model.evaluate(test_images,test_labels)
# print("Restored model,accuracy:{5.2f}%".format(100*acc))

# checkpoint_path="cp-{epoch:04d}.ckpt"
# checkpoint_dir=os.path.dirname(checkpoint_path)
# cp_callback=tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path,verbose=1,save_weights_only=True,
#     period=5   #每5个时期保存一次唯一命名的检查点
# )
# model=create_model()
# model.fit(train_images,train_labels,epochs=50,callbacks=[cp_callback],
#           validation_data=(test_images,test_labels),
#           verbose=0)
#
#
# latest = tf.train.latest_checkpoint(checkpoint_dir) ###有点问题，获取为空
# print(latest)
# model.load_weights(latest)
# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

model=create_model()
model.fit(train_images,train_labels,epochs=50,
          validation_data=(test_images,test_labels),
          verbose=0)

model.save_weights('my_checkpoint')

# Restore the weights
model.load_weights('my_checkpoint')   #手动保存并加载模型

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))




