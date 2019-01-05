import sys
import warnings

from sklearn import ensemble
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

pd.set_option('display.max_columns', 500)
path = r"C:\Users\miair\PycharmProjects\untitled11\spam.csv" #путь к файлу spam.csv
data = pd.read_csv(path, skip_blank_lines =True, sep='delimiter') #чтение apsm.csv
data['class'], data['comments'] = data['v1,v2,,,'].str.split(';', 1).str #CSV файл автоматически неделим, разделяем вручную
data['class'] = data['class'].replace(regex='".{3,}', value='') #убираем кривые классы
data_train = data.dropna(axis=0) #удаляем все пустые значения

spam = preprocessing.LabelEncoder() #переводим категориальный признак в числовой(spam, no spam)
spam.fit(data['class'])
data['class-number'] = spam.transform(data['class']) #записываем числовой признак в новый столбец
print('Choose algorithm text analysis(Bags-of-words,tf-idf or n-grams): ') #выбор алгоритма анализа текстовых признаков
Name_text_analysis = input() #название алгоритма анализа тестовых признаков
print('Choose ML algorithm(LogisticRegression, RandomForest, ComplementNB, Perceptron): ') #выбор алгоритма машинного обучения
Name_algorithm_ML = input() #название алгоритма машинного обучения
print('enter text (text/N):')  # введите текст
text = input()  # текст
U = text
text = [text]
if (Name_text_analysis == 'Bags-of-words'):
    Bagsofwords = CountVectorizer().fit(data['comments'].values.astype('U'))
    training_scores = Bagsofwords.transform(data['comments'].values.astype('U')) #преобразование теста в массив числовых признаков с помощью алгоритма "мешок слов"
    text = Bagsofwords.transform(text)
elif (Name_text_analysis == 'n-grams'):
    ngrams = CountVectorizer(min_df = 4, ngram_range=(1, 2)).fit(data['comments'].values.astype('U'))
    training_scores = ngrams.transform(data['comments'].values.astype('U')) #преобразование теста в массив числовых признаков с помощью алгоритма "n-grams"
    text = ngrams.transform(text)
elif (Name_text_analysis == 'tf-idf'):
    tf_idf = TfidfVectorizer()
    training_scores = tf_idf.fit_transform(data['comments'].values.astype('U')) #преобразование теста в массив числовых признаков с помощью алгоритма "tf-idf"
    text = tf_idf.transform(text)
else:
    print('Error')
    sys.exit()

X_train, X_test, y_train, y_test = train_test_split(training_scores, data['class-number'], test_size = 0.3, random_state = 11) #делим набор данных на тестовую и тренировочную
if (Name_algorithm_ML == 'LogisticRegression'):
    model = LogisticRegression() #Логистическая регрессия
    model.fit(X_train, y_train)#обучение модели
    predictions = model.predict(X_train) #предсказание обучающего набора
    predictions_test = model.predict(X_test) #предсказание тестового набора
    predict_text = model.predict(text)
elif(Name_algorithm_ML == 'RandomForest'):
    parameter_space = {'bootstrap': [True, False],
                       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_leaf': [1, 2, 4],
                       'min_samples_split': [2, 5, 10],
                       'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    model = ensemble.RandomForestClassifier() #Случайный лес
    clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)#обучение модели
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
elif(Name_algorithm_ML == 'ComplementNB'):
    model = ComplementNB() #Наивный баесовский классификатор
    model.fit(X_train, y_train)#обучение модели
    predictions = model.predict(X_train) #предсказание обучающего набора
    predictions_test = model.predict(X_test) #предсказание тестового набора
    predict_text = model.predict(text)

elif(Name_algorithm_ML == 'Perceptron'):
    neuronov_v_sloe = []  # массив количествва нейроннов в скрытом слое
    print('comparison (Y/N):')  #выбор будет ли произведен вычисления слоев перцептронов
    comparison = input()
    if (comparison == 'Y'):
        neuronov = [1, 5, 50, 100, 300, 500, 1000]
        for i in range(7):
            for j in range(neuronov[i]):
                neuronov_v_sloe.append(100) # добавление количества нейронов в скрытом слое
            model = MLPClassifier(hidden_layer_sizes=neuronov_v_sloe).fit(X_train, y_train)  # Многослойный перцептрон
            predictions = model.predict(X_train) #предсказание обучающего набора
            predictions_test = model.predict(X_test) #предсказание тестового набора
            err_train = f1_score(y_train, predictions, average='micro')  # F-мера на обучающем наборе
            err_test = f1_score(y_test, predictions_test, average='micro')  # F-мера на тестовом наборе
            print('F-мера обучающей выборки:', err_train)  #F-мера на обучающем наборе
            print('F-мера тестовой выборки:', err_test) #F-мера на тестовом наборе
            print('Количество слоев в нейронной сети:',model.n_layers_) #выводит колинчество слоев сети
            neuronov_v_sloe = []
    elif (comparison == 'N'):
        for i in range(10):
            neuronov_v_sloe.append(20)  # добавление количества нейронов в скрытом слое
        model = MLPClassifier(hidden_layer_sizes=neuronov_v_sloe).fit(X_train, y_train)  # Многослойный перцептрон
        predict_text = model.predict(text)
        predictions = model.predict(X_train)  # предсказание обучающего набора
        predictions_test = model.predict(X_test)  # предсказание тестового набора


err_train = f1_score(y_train, predictions, average='micro') #F-мера на обучающем наборе
err_test = f1_score(y_test, predictions_test, average='micro') #F-мера на тестовом наборе
print('F-мера обучающей выборки:', err_train) #вывод результатов
print('F-мера тестовой выборки:', err_test)
if (U != 'N'):
    if predict_text[0]==1:
        print('Предсказание классификатора: спам', predict_text)
    if predict_text[0] == 0:
        print('Предсказание классификатора: не спам', predict_text)
