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

def data_preparation_no_split(features, nbits):


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
    
def logp_predict(Smi, Path_to_the_model):
  mfp_features = mfp_generator(Smi, 4, 2048)
  MACCS_features = MACCS_generator(Smi)
  features = concat_features(mfp_features, MACCS_features)
  Target= data_preparation_no_split(features, nbits)
  OWPCP = tf.keras.models.load_model(Path_to_the_model) 
  Predicted_logP = OWPCP.predict(Target)
  print(Predicted_logP)

    return X_tr
