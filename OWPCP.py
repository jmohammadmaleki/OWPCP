from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import rdkit
import keras_tuner as kt
from rdkit import Chem
from tensorflow.keras.activations import linear, relu, elu, tanh
from sklearn.model_selection import train_test_split
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import sys
import keras
import keras.backend as K
import tensorflow as tf
from __future__ import print_function, division
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from keras import backend as backend


def mfp_generator(smiles_list, radius, nbits):
    # Pre-allocate the array for Morgan Fingerprints (nbits columns)
    mfp_features = np.zeros((len(smiles_list), nbits), dtype=int)

    # Loop over the SMILES and generate fingerprints
    for idx, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol:  # Ensure the molecule is valid
            fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
            mfp_vec = fpgen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(mfp_vec, mfp_features[idx])
    
    return mfp_features

def MACCS_generator(smiles_list):
    # Pre-allocate the array for MACCS fingerprints (167 columns)
    MACCS_features = np.zeros((len(smiles_list), 167), dtype=int)

    # Loop over the SMILES and generate MACCS fingerprints
    for idx, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol:  # Ensure the molecule is valid
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            DataStructs.ConvertToNumpyArray(maccs_fp, MACCS_features[idx])

    return MACCS_features

def concat_features(feature_1, feature_2):

  features = np.append(feature_1, feature_2, axis=1)

  return(features)

def data_preparation(features, train_size, nbits):

    X_train, X_test, y_train, y_test = train_test_split(features, target_values, test_size=(1-train_size), random_state=0)

    X_tr = []
    X_tst = []

    mfp1 = []
    MACCS1 = []

    for i in X_train:
        temp_mfp1 = i[:nbits]
        mfp1.append(temp_mfp1)
        temp_MACCS1 = i[nbits:]
        MACCS1.append(temp_MACCS1)
    mfp1 = np.asarray(mfp1)
    MACCS1 = np.asarray(MACCS1)

    X_tr.append(mfp1)
    X_tr.append(MACCS1)

    mfp1 = []
    MACCS1 = []

    for i in X_test:
        temp_mfp1 = i[:nbits]
        mfp1.append(temp_mfp1)
        temp_MACCS1 = i[nbits:]
        MACCS1.append(temp_MACCS1)
    mfp1 = np.asarray(mfp1)
    MACCS1 = np.asarray(MACCS1)

    X_tst.append(mfp1)
    X_tst.append(MACCS1)

    return X_tr,y_train, X_tst, y_test

def data_preparation_no_split(features, target_values, nbits):

    y_train = target_values
    X_tr = []
    X_tst = []

    mfp1 = []
    MACCS1 = []

    for i in features:
        temp_mfp1 = i[:nbits]
        mfp1.append(temp_mfp1)
        temp_MACCS1 = i[nbits:]
        MACCS1.append(temp_MACCS1)
    mfp1 = np.asarray(mfp1)
    MACCS1 = np.asarray(MACCS1)

    X_tr.append(mfp1)
    X_tr.append(MACCS1)


    return X_tr,y_train

def logp_pred(Smi, Path_to_the_model):
  mfp_features = mfp_generator(Smi, 4, 2048)
  MACCS_features = MACCS_generator(Smi)
  features = concat_features(mfp_features, MACCS_features)
  Target= data_preparation_no_split(features, nbits)
  OWPCP = tf.keras.models.load_model(Path_to_the_model) 
  Predicted_logP = OWPCP.predict(Target)
  print(f'Predicted logP for {Smi[0]} is {Predicted_logP[0][0]}')


### Implementation of the OWPCP model ###
def model_builder(hp):
    MFP_l1_units = hp.Int('units_1', min_value = 512, max_value = 4096, step =258 )
    MFP_l2_units = hp.Int('units_2', min_value = 512, max_value = 4096, step =258 )
    MACCSK_l1_units = hp.Int('units_3', min_value = 512, max_value = 4096, step =258 )
    MACCSK_l2_units = hp.Int('units_4', min_value = 512, max_value = 4096, step =258 )
    out_l1_units = hp.Int('units_5', min_value = 512, max_value = 4096, step =258 )
    out_l2_units = hp.Int('units_6', min_value = 512, max_value = 4096, step =258 )
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    MFP_l1_activation = hp.Choice('activation_1', values = ['relu', 'elu', 'linear', 'tanh'] )
    MFP_l2_activation = hp.Choice('activation_2', values = ['relu', 'elu', 'linear', 'tanh'] )
    MACCSK_l1_activation = hp.Choice('activation_3', values = ['relu', 'elu', 'linear', 'tanh'] )
    MACCSK_l2_activation = hp.Choice('activation_4', values = ['relu', 'elu', 'linear', 'tanh'] )
    out_l1_activation = hp.Choice('activation_5', values = ['relu', 'elu', 'linear', 'tanh'] )
    out_l2_activation = hp.Choice('activation_6', values = ['relu', 'elu', 'linear', 'tanh'] )
    MFP_drop = hp.Float('drop_1', min_value = 0, max_value = 0.5, step = 0.05 )
    MACCSK_drop = hp.Float('drop_2', min_value = 0, max_value = 0.5, step = 0.05 )
    out_drop = hp.Float('drop_3', min_value = 0, max_value = 0.5, step = 0.05 )

    #Encoder for Morgan Fingerprints
    MfingerP_vec = Input(shape=(nbits,))
    MFP = Dense(MFP_l1_units, activation=MFP_l1_activation, kernel_initializer='he_normal')(MfingerP_vec)
    MFP = Dropout(MFP_drop)(MFP)
    out_MFP1 = Dense(MFP_l2_units, activation=MFP_l2_activation)(MFP)
    model_MFP = Model(MfingerP_vec, out_MFP1)

    MFP_inp = Input(shape=(nbits,))
    out_MFP = model_MFP(MFP_inp)

    #Encoder for MACCS keys
    MACCSK_vec = Input(shape=(167,))
    MACCSK1 = Dense(MACCSK_l1_units, activation=MACCSK_l1_activation, kernel_initializer='he_normal')(MACCSK_vec)
    MACCSK1 = Dropout(MACCSK_drop)(MACCSK1)
    out_MACCS_1 = Dense(MACCSK_l2_units, activation = MACCSK_l2_activation)(MACCSK1)
    model_MACCSK = Model(MACCSK_vec, out_MACCS_1)

    MACCSK_inp = Input(shape=(167,))
    out_MACCSK = model_MACCSK(MACCSK_inp)

    #Decoder to predict the octanol water prtition coeffiecient
    concatenated_MFP = keras.layers.concatenate([out_MACCSK, out_MFP])
    out_c1 = Dense(out_l1_units, activation=out_l1_activation)(concatenated_MFP)
    out_c1 = Dropout(out_drop)(out_c1)
    out_c1 = Dense(out_l2_units, activation=out_l2_activation)(out_c1)
    out_c1 = Dense(1, activation='linear', name="Predictor_LogP")(out_c1)

    p_model = Model(inputs= [MFP_inp, MACCSK_inp], outputs =[out_c1])

    p_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=float(hp_learning_rate),
                                                    beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                                                    loss={'Predictor_LogP': 'mse'},
                                                    metrics={'Predictor_LogP': 'mse'})


    return p_model

    
def OWPCP(X_tr, Y_tr, nbits):
    #Encoder for mfp
    MfingerP_vec = Input(shape=(nbits,))
    MFP = Dense(int(best_hps.get('units_1')), activation=best_hps.get('activation_1'), kernel_initializer='he_normal')(MfingerP_vec)
    MFP = Dropout(best_hps.get('drop_1'))(MFP)
    out_MFP1 = Dense(int(best_hps.get('units_2')), activation=best_hps.get('activation_2'))(MFP)
    model_MFP = Model(MfingerP_vec, out_MFP1)

    MFP_inp = Input(shape=(nbits,))
    out_MFP = model_MFP(MFP_inp)

    #Encoder for MACCS keys
    MACCSK_vec = Input(shape=(167,))
    MACCSK1 = Dense(int(best_hps.get('units_3')), activation=best_hps.get('activation_3'), kernel_initializer='he_normal')(MACCSK_vec)
    MACCSK1 = Dropout(best_hps.get('drop_2'))(MACCSK1)
    out_MACCS_1 = Dense(int(best_hps.get('units_4')), activation = best_hps.get('activation_4'))(MACCSK1)
    model_MACCSK = Model(MACCSK_vec, out_MACCS_1)

    MACCSK_inp = Input(shape=(167,))
    out_MACCSK = model_MACCSK(MACCSK_inp)

    #Decoder to predict the synergy score and the single drug response of each drug
    concatenated_MFP = keras.layers.concatenate([out_MACCSK, out_MFP])
    out_c1 = Dense(int(best_hps.get('units_5')), activation=best_hps.get('activation_5'))(concatenated_MFP)
    out_c1 = Dropout(best_hps.get('drop_3'))(out_c1)
    out_c1 = Dense(int(best_hps.get('units_6')), activation=best_hps.get('activation_6'))(out_c1)
    out_c1 = Dense(1, activation='linear', name="Predictor_LogP")(out_c1)

    D_model = Model(inputs= [MFP_inp, MACCSK_inp], outputs =[out_c1])

    D_model.compile(optimizer= tf.keras.optimizers.Adamax(learning_rate=float(best_hps.get('learning_rate')),
                                                    beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                                                    loss={'Predictor_LogP': 'mse'},
                                                    metrics=['mse', 'mae', keras.metrics.RootMeanSquaredError(), tf.keras.metrics.R2Score()])

    model_checkpoint = ModelCheckpoint('best_model.h5.keras', monitor='val_loss', save_best_only=True, mode='min')

    history = D_model.fit(X_tr, Y_tr, batch_size=128, epochs=100, verbose=1,
                                     validation_split=0.2, callbacks=[model_checkpoint])

    return D_model, history



#Read, Load, and prepare data
data = pd.read_csv('Full_lib.csv')
smiles_list = data['Smiles'].tolist()
LogP_list = data['logP'].tolist()
target_values = np.array(LogP_list)
target_values = target_values.reshape(len(LogP_list),1)


radius = 4
nbits = 2048
train_size = 0.9
mfp_features = mfp_generator(smiles_list, radius, nbits)
MACCS_features = MACCS_generator(smiles_list)
features = concat_features(mfp_features, MACCS_features)
training_set,y_train, testing_set, y_test = data_preparation(features, train_size, nbits)

#Tune the HPs
tuner = kt.Hyperband(model_builder,
                     objective = "val_mse",
                     max_epochs = 10,
                     factor = 3,
                     directory ='/content',
                     project_name = 'model')
stop_early = tf.keras.callbacks.EarlyStopping(monitor = "val_mse", patience = 5)
tuner.search(training_set, y_train, epochs = 20, batch_size = 256, validation_split = 0.111, callbacks = [stop_early])


#Train the model with Tuned HPs
epochs = 100
trained_MOWPCP, history = OWPCP(training_set, y_train, nbits, epochs)
