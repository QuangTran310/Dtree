import pandas as pd
import pydotplus as py
import matplotlib.pyplot as pl
from sklearn.tree import DecisionTreeClassifier as dtree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
from sklearn import tree

def read_csv():
    data = pd.read_csv('C:\\Users\\tmquang\\Downloads\\data.csv')
    return data


def set_values():
    global classifier
    data = read_csv()
    heat = {'YES': 1, 'NO': 0}
    wet = {'YES': 1, 'NO': 0}
    data['heat'] = data['heat'].map(heat)
    data['wet'] = data['wet'].map(wet)

    ind_vals = ['Day', 'Month', 'Year'
        , 'mean_temp', 'max_temp', 'min_temp'
        , 'meanhum', 'meandew', 'pressure']
    x = data[ind_vals]
    d_vals = ['wet']
    y = data[d_vals]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1, shuffle=True)

    classifier = dtree(criterion='entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    accuracy = classifier.score(x_test, y_test)
    print(y_pred)
    print(accuracy)
    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))
    print('')
    text_representation = tree.export_text(classifier)
    print(text_representation)


set_values()


def fig():
    fig = pl.figure(figsize=(25, 20))
    _ = tree.plot_tree(classifier)
    fig.savefig("decistion_tree.png")
fig()
