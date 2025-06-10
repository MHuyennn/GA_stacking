from __init__ import *

def MobileNet_Model(img_size=299, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 1))
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(BatchNormalization())
        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))  # Sử dụng NUM_CLASSES
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if isSummary:
            print(model.summary())
    return model

def ResNet50_Model(img_size=299, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 1))
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(BatchNormalization())
        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))  # Sử dụng NUM_CLASSES
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if isSummary:
            print(model.summary())
    return model

def VGG16_Model(img_size=299, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 1))
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(BatchNormalization())
        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))  # Sử dụng NUM_CLASSES
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if isSummary:
            print(model.summary())
    return model

def Xception_Model(img_size=299, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 1))
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(BatchNormalization())
        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))  # Sử dụng NUM_CLASSES
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if isSummary:
            print(model.summary())
    return model

def LGBM_Model():
    return LGBMClassifier(objective='multiclass', num_class=NUM_CLASSES)

def XGB_Model():
    return XGBClassifier(objective='multi:softprob', num_class=NUM_CLASSES)

def CatBoost_Model():
    return CatBoostClassifier(verbose=0, loss_function='MultiClass', classes_count=NUM_CLASSES)

def LogisticRegression_Model():
    return LogisticRegression(multi_class='multinomial')

def RandomForestClassifier_Model():
    return RandomForestClassifier()

def AdaBoostClassifier_Model():
    return AdaBoostClassifier()

def KNN_Model():
    return KNeighborsClassifier()

MODEL_FACTORY = {
    'VGG16': VGG16_Model, 'Xception': Xception_Model, 'ResNet50': ResNet50_Model, 'MobileNet': MobileNet_Model,
    'LogisticRegression': LogisticRegression_Model, 'RandomForestClassifier': RandomForestClassifier_Model,
    'KNN': KNN_Model, 'AdaBoostClassifier': AdaBoostClassifier_Model, 'LGBM': LGBM_Model, 'XGB': XGB_Model, 'CatBoost': CatBoost_Model
}
