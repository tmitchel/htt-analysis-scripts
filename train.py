from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


training_variables = [
    'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2',
    'costhetastar', 'mjj', 'higgs_pT', 'm_sv'
]


def build_model(nvars: int) -> Sequential:
    model = Sequential()
    model.add(Dense(nvars*2, input_shape=(nvars,), name='input', activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(nvars, name='hidden', activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, name='output', activation='sigmoid', kernel_initializer='normal'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_callbacks(model_name: str) -> list:
    return [
        EarlyStopping(monitor='val_loss', patience=50),
        ModelCheckpoint(f'Output/models/{model_name}.hdf5', monitor='val_loss',
                        verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto',
                        period=1
                        ),
        # TensorBoard(log_dir=f"logs/{time()}", histogram_freq=200, write_grads=False, write_images=True)
    ]


def confusion_matrix(data, label, prediction, threshold=0.5):
    result = pd.DataFrame(data, columns=training_variables)
    result['label'] = label
    result['predict'] = prediction

    true_pos = len(result[(result.label == 1) & (result.predict >= threshold)])
    true_neg = len(result[(result.label == 0) & (result.predict < threshold)])
    false_pos = len(result[(result.label == 1) & (result.predict < threshold)])
    false_neg = len(result[(result.label == 0) & (result.predict >= threshold)])

    return pd.DataFrame([[true_pos, false_neg], [false_pos, true_neg]], index=['signal', 'background'], columns=['pred >= 0.5', 'pred < 0.5'])

def main(args):
    nvars = len(training_variables)
    model = build_model(nvars)
    callbacks = build_callbacks(args.model)

    # load training data
    input_data = pd.HDFStore(args.input)
    sig = input_data.get(args.signal)
    bkg = input_data.get(args.background)
    scaler_info = input_data.get('standardization')

    # apply VBF selection
    sig = sig[(sig.is_signal > 0) & (sig.njets > 1) & (sig.mjj > 300)]
    bkg = bkg[(bkg.is_signal > 0) & (bkg.njets > 1) & (bkg.mjj > 300)]
    print(f'Unscaled No. Signal Events {len(sig)}')
    print(f'Unscaled No. Background Events {len(bkg)}')

    # scale signals to same number of events
    scaleto = max(len(sig), len(bkg))
    sig.loc[:, 'scaled_evtwt'] = sig.loc[:, 'scaled_evtwt'] * scaleto / len(sig)
    bkg.loc[:, 'scaled_evtwt'] = bkg.loc[:, 'scaled_evtwt'] * scaleto / len(bkg)

    # build scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_info['mean'].values.reshape(1, -1)
    scaler.scale_ = scaler_info['scale'].values.reshape(1, -1)
    scaler.var_ = scaler_info['variance'].values.reshape(1, -1)
    scaler.n_samples_seen_ = scaler_info['nsamples'].values.reshape(1, -1)
    scaler_columns = scaler_info.index.values

    # setup training dataframe
    dataset = pd.concat([sig, bkg])
    for var in training_variables:
        dataset[var] -= scaler_info.loc[var, 'mean']
        dataset[var] /= scaler_info.loc[var, 'scale']
    print(dataset[training_variables])

   # split into testing and training sets
    train_data, test_data, train_labels, test_labels, train_weights, _ = train_test_split(
        dataset[training_variables].values, dataset.signalLabel.values, dataset.scaled_evtwt.values,
        test_size=0.05, random_state=7
    )

    history = model.fit(train_data, train_labels, shuffle=True,
                        epochs=10000, batch_size=1024, verbose=True,
                        callbacks=callbacks, validation_split=0.25, sample_weight=train_weights
                        )

    if not args.dont_plot:
        # plot loss vs epoch
        ax = plt.subplot(2, 1, 1)
        ax.plot(history.history['loss'], label='loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.legend(loc="upper right")
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')

        # plot accuracy vs epoch
        ax = plt.subplot(2, 1, 2)
        ax.plot(history.history['accuracy'], label='acc')
        ax.plot(history.history['val_accuracy'], label='val_acc')
        ax.legend(loc="upper left")
        ax.set_xlabel('epoch')
        ax.set_ylabel('acc')
        plt.savefig(f'Output/plots/{args.model}.png')
        plt.clf()

        # testing confusion matrix
        test_pred = model.predict(test_data)
        test_conf_matrix = confusion_matrix(test_data, test_labels, test_pred)
        print(test_conf_matrix)
        ax = sns.heatmap(test_conf_matrix, annot=True, fmt="d").get_figure()
        ax.savefig(f'Output/plots/{args.model}-test-conf.png')
        plt.clf()

        # training confusion matrix
        train_pred = model.predict(train_data)
        train_conf_matrix = confusion_matrix(train_data, train_labels, train_pred)
        print(train_conf_matrix)
        ax = sns.heatmap(train_conf_matrix, annot=True, fmt="d").get_figure()
        ax.savefig(f'Output/plots/{args.model}-train-conf.png')
        plt.clf()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', required=True, help='name of the model to train')
    parser.add_argument('--input', '-i', required=True, help='full name of input file')
    parser.add_argument('--signal', '-s', required=True, help='name of signal file')
    parser.add_argument('--background', '-b', required=True, help='name of background file')
    parser.add_argument('--dont-plot', action='store_true', dest='dont_plot', help='don\'t make training plots')
    main(parser.parse_args())
