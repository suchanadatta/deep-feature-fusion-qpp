import sys, os, random, argparse
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Lambda
from keras.layers import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import accuracy_score
import eval_rank_corr as eval

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

class PointInstance:
    def __init__(self, id):
        self.qid_a = id

    def __str__(self):
        return "({})".format(self.qid_a)

    def getKey(self):
        return "{}".format(self.qid_a)

class PointInstanceIds:
    def __init__(self, testSet):
        self.data = {}
        for id in testSet:
            instance = PointInstance(id)
            self.data[instance.getKey()] = instance

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
    def __init__(self, paired_instances_ids, dataFolder, batch_size, dim, dim_label):
        self.paired_instances_ids = paired_instances_ids
        self.dim_hist = [dim, dim, dim, dim, dim_label]
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
        X = [np.empty((self.batch_size, *self.dim_hist[i])) for i in range(5)]
        Z = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            f_dim = a_data.matrix.shape[1]
            #print ("f_dim:{}".format(f_dim))
            num_row = a_data.matrix.shape[0]

            a_data_top = a_data.matrix[0:10, :]
            a_data_top = np.resize(a_data_top, (10, f_dim))

            a_data_bottom = a_data.matrix[num_row-10:, :]
            #a_data_bottom = a_data.matrix[-10:, :]
            a_data_bottom = np.resize(a_data_bottom, (10, f_dim))
            #a_feature = np.asarray(self.qid_features.get(int(a_id)))
            #a_feature = np.pad(a_feature, (0, 10 * 120 * 1 - 78), 'constant')
            #print ("__data_generation-a_data_top:{}:{}".format(a_data_top.shape, a_id))
            #print ("__data_generation-a_data_bottom:{}:{}".format(a_data_bottom.shape, a_id))
            assert a_data_bottom.shape == (10,f_dim), a_id

            b_data = InteractionData(b_id, self.dataDir)
            num_row = b_data.matrix.shape[0]
            b_data_top = b_data.matrix[0:10, :]
            b_data_top = np.resize(b_data_top, (10, f_dim))

            b_data_bottom = b_data.matrix[num_row-10:, :]
            #b_data_bottom = b_data.matrix[-10:, :]
            b_data_bottom = np.resize(b_data_bottom, (10, f_dim))

            #b_feature = np.asarray(self.qid_features.get(int(b_id)))
            #b_feature = np.pad(b_feature, (0, 10 * 120 * 1 - 78), 'constant')
            #print ("__data_generation-b_data_top:{}:{}".format(b_data_top.shape, b_id))
            #print ("__data_generation-b_data_bottom:{}:{}".format(b_data_bottom.shape, b_id))
            assert b_data_bottom.shape == (10,f_dim), b_id

            w_top, h_top = a_data_top.shape
            w_bottom, h_bottom = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top, h_top, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom, h_bottom, 1)
            #a_feature = a_feature.reshape(w_top, h_top, 1)
            b_data_top = b_data_top.reshape(w_top, h_top, 1)
            b_data_bottom = b_data_bottom.reshape(w_bottom, h_bottom, 1)
            #b_feature = b_feature.reshape(w_top, h_top, 1)

            X[0][i,] = a_data_top
            X[1][i,] = a_data_bottom
            X[2][i,] = b_data_top
            X[3][i,] = b_data_bottom
            X[4][i,] = paired_instance.class_label
            
            Z[i] = paired_instance.class_label

        return X, Z

class PairCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder, batch_size, dim):
        self.paired_instances_ids = paired_instances_ids
        self.dim_hist = [dim] * 4
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
        X = [np.empty((self.batch_size, *self.dim_hist[i])) for i in range(4)]

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            f_dim = a_data.matrix.shape[1]
            num_row = a_data.matrix.shape[0]

            a_data_top = a_data.matrix[0:10, :]
            a_data_top = np.resize(a_data_top, (10, f_dim))

            a_data_bottom = a_data.matrix[num_row-10:,:]
            #a_data_bottom = a_data.matrix[-10:, :]
            a_data_bottom = np.resize(a_data_bottom, (10, f_dim))

            #a_feature = np.asarray(self.qid_features.get(int(a_id)))
            #a_feature = np.pad(a_feature, (0, 10 * 120 * 1 - 78), 'constant')
            assert a_data_bottom.shape == (10, f_dim), a_id

            b_data = InteractionData(b_id, self.dataDir)
            num_row = b_data.matrix.shape[0]
            b_data_top = b_data.matrix[0:10, :]
            b_data_top = np.resize(b_data_top, (10, f_dim))

            b_data_bottom = b_data.matrix[num_row-10:, :]
            #b_data_bottom = b_data.matrix[-10:, :]
            b_data_bottom = np.resize(b_data_bottom, (10, f_dim))

            #b_feature = np.asarray(self.qid_features.get(int(b_id)))
            #b_feature = np.pad(b_feature, (0, 10 * 120 * 1 - 78), 'constant')
            assert b_data_bottom.shape == (10, f_dim), b_id

            w_top, h_top = a_data_top.shape
            w_bottom, h_bottom = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top, h_top, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom, h_bottom, 1)
            #a_feature = a_feature.reshape(w_top, h_top, 1)
            b_data_top = b_data_top.reshape(w_top, h_top, 1)
            b_data_bottom = b_data_bottom.reshape(w_bottom, h_bottom, 1)
            #b_feature = b_feature.reshape(w_top, h_top, 1)

            X[0][i,] = a_data_top
            X[1][i,] = a_data_bottom
            X[2][i,] = b_data_top
            X[3][i,] = b_data_bottom
   
        return X

class PointCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder, batch_size, dim_top, dim_bottom, topDocs,
                 bottomDocs, interMatrix):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = [dim_top, dim_bottom]
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.K = topDocs
        self.L = bottomDocs
        self.M = interMatrix
        self.on_epoch_end()
        self.totalRetDocs = 100

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim[i])) for i in range(1)]

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            a_data_top = a_data.matrix[0:self.K, :]
            a_data_bottom = a_data.matrix[(self.totalRetDocs - self.L):, :]

            w_top_a, h_top_a = a_data_top.shape
            w_bottom_a, h_bottom_a = a_data_bottom.shape
            a_data_top = a_data_top.reshape(w_top_a, h_top_a, 1)
            a_data_bottom = a_data_bottom.reshape(w_bottom_a, h_bottom_a, 1)

            X[0][i,] = a_data_top
            # X[1][i,] = a_data_bottom

        return X    
    
class ConvModel:

    def pair_loss(x):
        # Pair Loss function.
        query1, query2, label = x
        hinge_margin = 1
        #keras.backend.print_tensor(query1)
        max_margin_hinge = hinge_margin - label * (query1 - query2)
        loss = keras.backend.maximum(0.0, max_margin_hinge)
        return loss

    def identity_loss(y_true, y_pred):
        return keras.backend.mean(y_pred)

    def base_model(input_shape):
        matrix_encoder = Sequential(name='sequence')
        matrix_encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
        # matrix_encoder.add(Dense(500))
        matrix_encoder.add(MaxPooling2D(padding='same'))
        matrix_encoder.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        # matrix_encoder.add(Dense(500))
        matrix_encoder.add(MaxPooling2D(padding='same'))
        matrix_encoder.add(Flatten())
        matrix_encoder.add(Dropout(0.2))
        matrix_encoder.add(Dense(128, activation='relu'))
        matrix_encoder.add(Dense(1, activation='sigmoid'))
        return matrix_encoder
        
    def build_siamese_custom_loss(input_shape_top, input_shape_bottom, input_label_shape, base_model):
        input_a_top = Input(shape=input_shape_top, dtype='float32')
        input_a_bottom = Input(shape=input_shape_bottom, dtype='float32')
        #a_feature = Input(shape=input_shape_top, dtype='float32')
        # input_c_top = Input(shape=input_label_shape, dtype='float32')

        input_b_top = Input(shape=input_shape_top, dtype='float32')
        input_b_bottom = Input(shape=input_shape_bottom, dtype='float32')
        input_c = Input(shape=input_label_shape, dtype='float32')
        #b_feature = Input(shape=input_shape_top, dtype='float32')

        encoded_a_top = base_model(input_a_top)
        encoded_a_bottom = base_model(input_a_bottom)
  
        merged_vector_a = concatenate([encoded_a_top, encoded_a_bottom], axis=-1, name='concatenate_1')
        
        encoded_b_top = base_model(input_b_top)
        encoded_b_bottom = base_model(input_b_bottom)

        merged_vector_b = concatenate([encoded_b_top, encoded_b_bottom], axis=-1, name='concatenate_2')

        #pair_indicator = Lambda(ConvModel.pair_loss)([activation_a_input, activation_b_input, input_c])
        pair_indicator = Lambda(ConvModel.pair_loss)([merged_vector_a, merged_vector_b, input_c])
        #pair_indicator = Lambda(ConvModel.pair_loss)([activation_a, activation_b])
        #prediction = Dense(1, activation='sigmoid')(pair_indicator)
        siamese_net_custom = Model(inputs=[input_a_top, input_a_bottom, input_b_top, input_b_bottom, 
                                           input_c], outputs=pair_indicator)
        return siamese_net_custom

def score_aggregate(predict_file, qpp_file):
    pred_confidence = pd.read_csv(predict_file, delimiter='\t')
    # print(pred_confidence)
    uniq_qids = list(set(pred_confidence['qid-a']))
    #print('Unique qids : ', uniq_qids)
    qid_qpp_dict = {}
    for qid in uniq_qids:
        sliced_df_qid_a = pred_confidence.loc[(pred_confidence['qid-a'] == qid), ['qid-a', 'qid-b', 'pred']]
        sliced_df_qid_b = pred_confidence.loc[(pred_confidence['qid-b'] == qid), ['qid-a', 'qid-b', 'pred']]
        qid_qpp_dict[qid] = round(float(sliced_df_qid_a['pred'].sum() / len(sliced_df_qid_a['pred'])), 5) + \
                            round(float(1 - sliced_df_qid_b['pred'].sum() / len(sliced_df_qid_b['pred'])), 5)
    #print('Pred QPP : ', qid_qpp_dict)

    # min-max normalize
    out_file = open(qpp_file, 'a')
    qpp_norm = ''
    for qid, qpp in qid_qpp_dict.items():
        norm = round((float(qpp) - min(qid_qpp_dict.values())) /
                     (max(qid_qpp_dict.values()) - min(qid_qpp_dict.values())), 5)
        qpp_norm += str(qid) + '\t' + str(norm) + '\n'
    out_file.write(qpp_norm)
    out_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-name',  default='interaction_hist') #per query ap100 scores
    parser.add_argument('--model-name',  default='pairwise_train_pairwise_inference') #per query ap100 scores
    parser.add_argument('--metric',  default='ap100') #per query ap100 scores
    parser.add_argument('--ap-path',  default='../data/per_query_ap100') #per query ap100 scores
    parser.add_argument('--train-hist', default='../data/interaction_letor_hist') # histograms for training query set
    parser.add_argument('--test-hist', default='../data/interaction_letor_hist') # histograms for test query set
    parser.add_argument('--train-ap-pairs', default='../data/train_pair_ap100_trec6_8_robust') #GT for training set <qid-a \t qid-b \t relativeAP(0/1)>
    parser.add_argument('--test-ap-pairs', default='../data/test_pair_ap100_trec7') # <qid-a \t qid-b> for test query pairs
    parser.add_argument('--test-ap-pairs-gt', default='../data/test_pair_gt_trec7')  # GT for test set <qid-a \t qid-b \t relativeAP(0/1)>
    parser.add_argument('--num-query-feature', default=37, type=int)
    parser.add_argument('--checkpoint', default='../checkpoint/model.weights')
    parser.add_argument('--prediction', default='../data/model.pred')
    parser.add_argument('--qpp-file-path', default='../data/qpp_ap100_predicted')
    parser.add_argument('--top-docs', default=10, type=int)
    parser.add_argument('--bottom-docs', default=10, type=int)
    parser.add_argument('--max-query-length', default=4, type=int)
    parser.add_argument('--num-channel', default=1, type=int)
    parser.add_argument('--bin-size', default=30, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    args = parser.parse_args()

    LR = 0.0001
    # create input paired instances (train/test)
    allPairs_train = PairedInstanceIds(idpairLabelsFile=args.train_ap_pairs) # <qid-a \t qid-b \t relativeAP>
    allPairsList_train = list(allPairs_train.data.values())
    allPairs_test = PairedInstanceIds(idpairLabelsFile=args.test_ap_pairs) # <qid-a \t qid-b>
    allPairsList_test = list(allPairs_test.data.values())
    print('{}/{} pairs for training'.format(len(allPairsList_train), len(allPairsList_train)))
    print('{}/{} pairs for testing'.format(len(allPairsList_test), len(allPairsList_test)))

    # create unique query list from the test split and GT for the same
    query_ap_dict = {}
    with open(args.ap_path, 'r') as f:
        content = f.readlines()
        for line in content:
            query_ap_dict[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

    # create unique query list from the test split
    uniq_test_set = set()
    with open(args.test_ap_pairs, 'r') as f:
        content = f.readlines()  
        for line in content:
            uniq_test_set.add(line.strip().split('\t')[0])

    # crete the GT for this test split
    test_ground_truth = []
    for qid in uniq_test_set:
        test_ground_truth.append(float(query_ap_dict.get(qid)))

    # create test generator(compatible to the model)
    all_point_test = PointInstanceIds(uniq_test_set)  # (4)
    all_point_list_test = list(all_point_test.data.values())
    
    base = ConvModel.base_model((args.top_docs, args.max_query_length * args.bin_size + args.num_query_feature, args.num_channel))
    
    siamese_model_custom = ConvModel.build_siamese_custom_loss((args.top_docs, args.max_query_length * args.bin_size + args.num_query_feature, args.num_channel), (args.bottom_docs, args.max_query_length * args.bin_size + args.num_query_feature, args.num_channel), (1, 1, 1), base)
    siamese_model_custom.compile(loss=ConvModel.identity_loss,
                     optimizer=keras.optimizers.Adam(LR),
                     metrics=['accuracy'])
    siamese_model_custom.summary()

    # train data generator
    training_generator = PairCmpDataGeneratorTrain(allPairsList_train, dataFolder=args.train_hist,
                        batch_size=args.batch_size,
                        dim=(args.top_docs, args.max_query_length * args.bin_size + args.num_query_feature, args.num_channel), dim_label=(1,1,1))
    print('Size of the training generator : ', len(training_generator))

    # learn model parameters with the train split
    siamese_model_custom.fit_generator(generator=training_generator, use_multiprocessing=True,
                           epochs=int(args.epochs), workers=4)
                            # validation_split=0.2,
                            # verbose=1)

    #siamese_model.save(args.checkpoint)
    test_generator = PointCmpDataGeneratorTest(all_point_list_test,
                    dataFolder=args.test_hist,
                    batch_size=1,
                    dim_top=(args.top_docs, args.max_query_length * args.bin_size + args.num_query_feature, args.num_channel),
                    dim_bottom=(args.top_docs, args.max_query_length * args.bin_size + args.num_query_feature, args.num_channel),
                    topDocs=args.top_docs, bottomDocs=args.bottom_docs,
                    interMatrix = args.max_query_length * args.bin_size + args.num_query_feature)
    print('Size of the test generator : ', len(test_generator))

    # make predictions
    predictions = base.predict(test_generator)
    print('Shape of the predicted matrix : ', predictions.shape)

    predict_list = []
    i = 0
    while i < len(predictions):
        predict_list.append(float(predictions[i]))
        i += 1

    #measure rank-correlations
    r, rho, tau = eval.reportRankCorr(test_ground_truth, predict_list)
    pearsons = r
    spearmans = rho
    kendalls = tau
    print('P-r = {:,.4f}, S-rho = {:.4f}, K-tau = {:.4f}'.format(pearsons, spearmans, kendalls))

if __name__ == '__main__':
    main()
