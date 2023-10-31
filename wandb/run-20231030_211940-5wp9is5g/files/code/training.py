import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import os
import wandb
from wandb.keras import WandbMetricsLogger
import time

ACTIONS = ["left", "right", "none"]
reshape = (-1, 16, 60)


def main():
    wandb.init(
        project="Directions_BCI"
    )
    # hpyerparameters
    epochs = 1
    batch_size = 32
    out_layers = [512, 512, 64]
    out_layers_activation = "relu"
    conv_activation = "relu"
    noise = 0

    if 'a' in wandb.config:
        out_layers = [wandb.config["a"], wandb.config["b"], wandb.config["c"]]

    wandb.config.update({
        "epochs": epochs,
        "batch_size": batch_size,
        "out_layers": out_layers,
        "out_layers_activation": out_layers_activation,
        "conv_activation": conv_activation,
        "noise": noise
    })

    def create_data(starting_dir="data"):
        training_data = {}
        for action in ACTIONS:
            if action not in training_data:
                training_data[action] = []

            data_dir = os.path.join(starting_dir, action)
            for item in os.listdir(data_dir):
                # print(action, item)
                data = np.load(os.path.join(data_dir, item))
                for item in data:
                    training_data[action].append(
                        item + np.random.normal(0, noise, item.shape))

        lengths = [len(training_data[action]) for action in ACTIONS]
        print(lengths)

        for action in ACTIONS:
            # note that regular shuffle is GOOF af
            np.random.shuffle(training_data[action])
            training_data[action] = training_data[action][:min(lengths)]

        lengths = [len(training_data[action]) for action in ACTIONS]
        print(lengths)
        # creating X, y
        combined_data = []
        for action in ACTIONS:
            for data in training_data[action][0::10]:

                if action == "left":
                    combined_data.append([data, [1, 0, 0]])

                elif action == "right":
                    combined_data.append([data, [0, 0, 1]])

                elif action == "none":
                    combined_data.append([data, [0, 1, 0]])

        np.random.shuffle(combined_data)
        print("length:", len(combined_data))
        return combined_data

    print("creating training data")
    traindata = create_data(starting_dir="data")
    train_X = []
    train_y = []
    for X, y in traindata:
        train_X.append(X)
        train_y.append(y)

    print("creating testing data")
    testdata = create_data(starting_dir="validation_data")
    test_X = []
    test_y = []
    for X, y in testdata:
        test_X.append(X)
        test_y.append(y)

    print(len(train_X))
    print(len(test_X))

    print(np.array(train_X).shape)
    train_X = np.array(train_X).reshape(reshape)
    test_X = np.array(test_X).reshape(reshape)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    model = Sequential()

    model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
    model.add(Activation(conv_activation))

    model.add(Conv1D(128, (2)))
    model.add(Activation(conv_activation))

    model.add(Conv1D(128, (2)))
    model.add(Activation(conv_activation))

    model.add(Conv1D(64, (2)))
    model.add(Activation(conv_activation))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(64, (2)))
    model.add(Activation(conv_activation))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Flatten())

    for layer in out_layers:
        model.add(Dense(layer))
        model.add(Activation(out_layers_activation))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_X, train_y, batch_size=batch_size,
              epochs=epochs, validation_data=(test_X, test_y), callbacks=[WandbMetricsLogger()])
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
    # print(score)
    MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{int(time.time())}-loss-{round(score[0],2)}.model"
    model.save(MODEL_NAME)
    print("saved:")
    print(MODEL_NAME)


"""
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "epoch/val_accuracy"},
    "parameters": {
        "a": {"values": [8, 16, 32, 64, 128, 256, 512, 1024]},
        "b": {"values": [8, 16, 32, 64, 128, 256, 512, 1024]},
        "c": {"values": [8, 16, 32, 64, 128, 256, 512, 1024]},
    },
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="Directions_BCI")

wandb.agent(sweep_id, function=main, count=10)
"""
main()
