import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# Загружаем данные, делим на фичи и таргет
data = pd.read_csv('training_mush.csv')
y_train = data['class']
X_train = data.drop('class', axis=1)

# Создаём классификатор
clf = RandomForestClassifier(random_state=0)
# Задаём параметры для перебора на GridSearchCV
params = {'n_estimators' : range(10, 51, 10), 'max_depth' : range(1, 13, 2), 'min_samples_leaf' : range(1, 8), 'min_samples_split' : range(2, 10, 2)}
# С помощью GridSearchCV находим классификатор с лучшими параметрами
GSCV = GridSearchCV(estimator=clf, param_grid=params, cv=3, n_jobs=-1, verbose=1)
GSCV.fit(X_train, y_train)
best_clf = GSCV.best_estimator_

# Строим график важности параметров
importance = pd.DataFrame(GSCV.best_estimator_.feature_importances_, index=X_train.columns, columns=['importance'])
importance.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()

# Загружаем тестовые данные, делаем предсказание
X_test = pd.read_csv('testing_mush.csv')
y_test = pd.read_csv('testing_y_mush.csv')
y_pred = best_clf.predict(X_test)

# Строим матрицу ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
graphic = sns.heatmap(conf_matrix, annot=True)
plt.show()

# Считаем ключевые метрики
print('Precision =', precision_score(y_test, y_pred),
      'Recall =', recall_score(y_test, y_pred),
      'F1-score =', f1_score(y_test, y_pred))
