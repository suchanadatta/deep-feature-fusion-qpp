import sys, os, random, argparse
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import accuracy_score

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

class InteractionData:
    # Interaction data of query qid with K top docs -
    # each row vector is a histogram of interaction data for a document
    def __init__(self, qid, dataPathBase):
        self.qid = qid
        histFile = "{}/{}.hist".format(dataPathBase, self.qid)
        histogram = np.genfromtxt(histFile, delimiter=" ")
        self.matrix = histogram[:, 4:]

class PairedInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid_a = l[0]
            self.qid_b = l[1]
            self.class_label = int(l[2])
        else:
            self.qid_a = l[0]
            self.qid_b = l[1]

    def __str__(self):
        return "({}, {})".format(self.qid_a, self.qid_b)

    def getKey(self):
        return "{}-{}".format(self.qid_a, self.qid_b)

# Separate instances for training/test sets etc. Load only the id pairs.
# Data is loaded later in batches with a subclass of Keras generator
class PairedInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpairLabelsFile):
        self.data = {}

        with open(idpairLabelsFile) as f:
            content = f.readlines()

        for x in content:
            instance = PairedInstance(x)
            self.data[instance.getKey()] = instance

class PairCmpDataGeneratorTrain(keras.utils.Sequence):
    def __init__(self, paired_instances_ids, qid_features, dataFolder, batch_size, dim):
        self.paired_instances_ids = paired_instances_ids
        self.qid_features = qid_features
        self.dim_hist = [dim] * 6
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs = [self.paired_instances_ids[k] for k in indexes]
        X = self.__data_generation(list_IDs)
        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        X = [np.empty((self.batch_size, *self.dim_hist[i])) for i in range(6)]
        Z = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            a_data_top = a_data.matrix[0:10, :]
            #a_data_bottom = a_data.matrix[90:, :]
            a_data_bottom = a_data.matrix[-10:, :]
            
            a_feature = np.asarray(self.qid_features.get(int(a_id)))
            a_feature = np.pad(a_feature, (0, 10 * 120 * 1 - 78), 'constant')
            assert a_data_bottom.shape == (10,120), a_id

            b_data = InteractionData(b_id, self.dataDir)
            b_data_top = b_data.matrix[0:10, :]
            #b_data_bottom = b_data.matrix[90:, :]
            b_data_bottom = b_data.matrix[-10:, :]
            b_feature = np.asarray(self.qid_features.get(int(b_id)))
            b_feature = np.pad(b_feature, (0, 10 * 120 * 1 - 78), 'constant')
            assert b_data_bottom.shape == (10,120), b_id

            w_top, h_top = a_data_top.shape
            w_bottom, h_bottom = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top, h_top, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom, h_bottom, 1)
            a_feature = a_feature.reshape(w_top, h_top, 1)
            b_data_top = b_data_top.reshape(w_top, h_top, 1)
            b_data_bottom = b_data_bottom.reshape(w_bottom, h_bottom, 1)
            b_feature = b_feature.reshape(w_top, h_top, 1)

            X[0][i,] = a_data_top
            X[1][i,] = a_data_bottom
            X[2][i,] = a_feature
            X[3][i,] = b_data_top
            X[4][i,] = b_data_bottom
            X[5][i,] = b_feature
            Z[i] = paired_instance.class_label

        return X, Z

class PairCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, qid_features, dataFolder, batch_size, dim):
        self.paired_instances_ids = paired_instances_ids
        self.qid_features = qid_features
        self.dim_hist = [dim] * 6
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs = [self.paired_instances_ids[k] for k in indexes]
        X = self.__data_generation(list_IDs)
        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        X = [np.empty((self.batch_size, *self.dim_hist[i])) for i in range(6)]

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            a_data_top = a_data.matrix[0:10, :]
            a_data_bottom = a_data.matrix[90:, :]
            a_feature = np.asarray(self.qid_features.get(int(a_id)))
            a_feature = np.pad(a_feature, (0, 10 * 120 * 1 - 78), 'constant')
            assert a_data_bottom.shape == (10, 120), a_id

            b_data = InteractionData(b_id, self.dataDir)
            b_data_top = b_data.matrix[0:10, :]
            b_data_bottom = b_data.matrix[90:, :]
            b_feature = np.asarray(self.qid_features.get(int(b_id)))
            b_feature = np.pad(b_feature, (0, 10 * 120 * 1 - 78), 'constant')
            assert b_data_bottom.shape == (10, 120), b_id

            w_top, h_top = a_data_top.shape
            w_bottom, h_bottom = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top, h_top, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom, h_bottom, 1)
            a_feature = a_feature.reshape(w_top, h_top, 1)
            b_data_top = b_data_top.reshape(w_top, h_top, 1)
            b_data_bottom = b_data_bottom.reshape(w_bottom, h_bottom, 1)
            b_feature = b_feature.reshape(w_top, h_top, 1)

            X[0][i,] = a_data_top
            X[1][i,] = a_data_bottom
            X[2][i,] = a_feature
            X[3][i,] = b_data_top
            X[4][i,] = b_data_bottom
            X[5][i,] = b_feature

        return X

def build_siamese(input_shape):
    # input for qid-a
    input_a_top = Input(shape=input_shape, dtype='float32')
    input_a_bottom = Input(shape=input_shape, dtype='float32')
    a_feature = Input(shape=input_shape, dtype='float32')
    # input for qid-b
    input_b_top = Input(shape=input_shape, dtype='float32')
    input_b_bottom = Input(shape=input_shape, dtype='float32')
    b_feature = Input(shape=input_shape, dtype='float32')

    # sequence-1
    matrix_encoder_init = Sequential(name='sequence_init')
    matrix_encoder_init.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    matrix_encoder_init.add(MaxPooling2D(padding='same'))
    matrix_encoder_init.add(Flatten())
    # sequence-2
    matrix_encoder_merge = Sequential(name='sequence_merge')
    matrix_encoder_merge.add(Dropout(0.2))
    matrix_encoder_merge.add(Dense(128, activation='relu'))
    # sequence-3
    matrix_encoder_flatten = Sequential(name='sequence_flat')
    matrix_encoder_flatten.add(Flatten())

    encoded_a_top = matrix_encoder_init(input_a_top)
    encoded_a_bottom = matrix_encoder_init(input_a_bottom)
    merged_vector_a = concatenate([encoded_a_top, encoded_a_bottom], axis=-1, name='concat_a')
    merged_vector_a_feature = concatenate([merged_vector_a, matrix_encoder_flatten(a_feature)], axis=-1, name='concat_a_feature')
    merged_vector_a_input = matrix_encoder_merge(merged_vector_a_feature)

    encoded_b_top = matrix_encoder_init(input_b_top)
    encoded_b_bottom = matrix_encoder_init(input_b_bottom)
    merged_vector_b = concatenate([encoded_b_top, encoded_b_bottom], axis=-1, name='concat_b')
    merged_vector_b_feature = concatenate([merged_vector_b, matrix_encoder_flatten(b_feature)], axis=-1, name='concat_b_feature')
    merged_vector_b_input = matrix_encoder_merge(merged_vector_b_feature)

    merged_vector = concatenate([merged_vector_a_input, merged_vector_b_input], axis=-1, name='concatenate_final')
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    siamese_net = Model([input_a_top, input_a_bottom, a_feature, input_b_top, input_b_bottom, b_feature],
                        outputs=predictions)
    return siamese_net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-hist', default='../data/test') # histograms for training query set
    parser.add_argument('--test-hist', default='../data/test') # histograms for test query set
    parser.add_argument('--train-ap-pairs', default='../data/train_pair') # GT for training set <qid-a \t qid-b \t relativeAP(0/1)>
    parser.add_argument('--test-ap-pairs', default='../data/test_pair') # <qid-a \t qid-b> for test query pairs
    parser.add_argument('--test-ap-pairs-gt', default='../data/test_pair.gt')  # GT for test set <qid-a \t qid-b \t relativeAP(0/1)>
    parser.add_argument('--query-feature', default='../data/feature/TREC-ROBUST_BM25_301_700.query_doc_aggregated_std_max.ftr')
    parser.add_argument('--checkpoint', default='../checkpoint/model.weights')
    parser.add_argument('--prediction', default='../data/model.pred')
    parser.add_argument('--top-docs', default=10, type=int)
    parser.add_argument('--bottom-docs', default=10, type=int)
    parser.add_argument('--max-query-length', default=4, type=int)
    parser.add_argument('--num-channel', default=1, type=int)
    parser.add_argument('--bin-size', default=30, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    args = parser.parse_args()

    # create input paired instances (train/test)
    allPairs_train = PairedInstanceIds(idpairLabelsFile=args.train_ap_pairs) # <qid-a \t qid-b \t relativeAP>
    allPairsList_train = list(allPairs_train.data.values())
    allPairs_test = PairedInstanceIds(idpairLabelsFile=args.test_ap_pairs) # <qid-a \t qid-b>
    allPairsList_test = list(allPairs_test.data.values())
    print('{}/{} pairs for training'.format(len(allPairsList_train), len(allPairsList_train)))
    print('{}/{} pairs for testing'.format(len(allPairsList_test), len(allPairsList_test)))

    # create additional feature dict
    features = pd.read_csv(args.query_feature, delimiter='\t')
    qid_features = features.set_index('QID').T.to_dict('list')

    # initialize siamese n/w
    siamese_model = build_siamese((args.top_docs, args.max_query_length * args.bin_size, 1))
    siamese_model.compile(loss = keras.losses.BinaryCrossentropy(),
                          optimizer = keras.optimizers.Adam(),
                          metrics=['accuracy'])
    siamese_model.summary()

    training_generator = PairCmpDataGeneratorTrain(allPairsList_train, qid_features, dataFolder=args.train_hist,
                                                   batch_size=args.batch_size,
                                                   dim=(args.top_docs, args.max_query_length * args.bin_size, args.num_channel))
    siamese_model.fit_generator(generator=training_generator,
                                use_multiprocessing=True,
                                epochs=args.epochs,
                                workers=4)
                                # validation_split=0.2,
                                # verbose=1)

    siamese_model.save_weights(args.checkpoint)
    test_generator = PairCmpDataGeneratorTest(allPairsList_test, qid_features, dataFolder=args.test_hist)
    predictions = siamese_model.predict(test_generator)
    # print('predict ::: ', predictions)
    # print('predict shape ::: ', predictions.shape)
    with open(args.prediction, 'w') as outFile:     # (9)
        i = 0
        for entry in test_generator.paired_instances_ids:
            if predictions[i][0] >= 0.5:
                outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '1\n')
            else:
                outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '0\n')
            i += 1
    outFile.close()

    # measure accuracy
    gt_file = np.genfromtxt(args.test_ap_pairs_gt, delimiter='\t')    # (10)
    actual = gt_file[:, 2:]
    # print(actual)
    predict_file = np.genfromtxt(args.prediction, delimiter='\t')
    predict = predict_file[:, 3:]
    # print(predict)
    score = accuracy_score(actual, predict)
    print('Accuracy : ', round(score, 4))

if __name__ == '__main__':
    main()
