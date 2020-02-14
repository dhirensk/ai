import tensorflow as tf
import pandas as pd
import numpy as np
print(tf.__version__)
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.head(5)
total_rows = dftrain['age'].count()
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(dftrain))

for idx, feature_batch in enumerate(titanic_slices.take(1)):
  for key, value in feature_batch.items():
    print("{}".format(value))




CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocab_list = dftrain[feature_name].unique()
    #print(vocab_list)
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab_list))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_fn(df, label, epochs,buffer_size,batch_size):
    def intput_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), label))
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size=batch_size).repeat(count=epochs)
        return dataset
    return intput_fn

test_input_fn = make_input_fn(dftrain,y_train,1,1000,16)()
for feature_batch, label in test_input_fn.take(1):
    print(list(feature_batch.keys()))
    print(list(feature_batch.values()))
    print(label)

train_input_fn = make_input_fn(dftrain,y_train,10,1000,16)
eval_input_fn = make_input_fn(dfeval,y_eval,1,500,16)

age_column = feature_columns[7]
tf.keras.layers.DenseFeatures(feature_columns[7])(feature_batch)

#DenseFeatures only accepts dense tensors, to inspect a categorical column you need to transform that to a indicator column first:

gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns)
classifier.train(input_fn=train_input_fn)

result = classifier.evaluate(input_fn=eval_input_fn)
print(result)
#{'accuracy': 0.7613636, 'accuracy_baseline': 0.625, 'auc': 0.83838385, 'auc_precision_recall': 0.78412384, 'average_loss': 0.47464514, 'label/mean': 0.375, 'loss': 0.48270804, 'precision': 0.68367344, 'prediction/mean': 0.3708848, 'recall': 0.67676765, 'global_step': 400}
