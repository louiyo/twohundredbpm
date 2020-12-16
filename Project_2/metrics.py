from keras import backend as K


def recall_m(y_true, y_pred):
    """
        the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of 
        false negatives. The recall is the ability of the classifier to find all the positive samples.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
        Calculate the ratio tp / (tp + fp), where tp is the number of true positives and fp the 
        number of false positives. The precision is the ability of the classifier not to label 
        as positive a sample that is negative. 
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
        Harmonic mean of the precision and recall.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
