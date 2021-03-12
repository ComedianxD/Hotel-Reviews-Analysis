import time
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def logistic_regression(x_tr, x_ts, y_tr, y_ts):
    print("\nLogistic Regression")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    model = LogisticRegression(multi_class='auto')

    print("Fitting Data...")
    trt_strt = time.time()
    model.fit(x_train, y_train)
    trt_end = time.time()
    print("Logistic Regression Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = model.predict(x_test)
    tst_end = time.time()
    print("Logistic Regression Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    print("Logistic Regression Accuracy", str(model.score(x_test, y_test) * 100) + ' %')
    print("Logistic Regression MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))


def svm_one_vs_rest_class(x_tr, x_ts, y_tr, y_ts):
    print("\nSVM One vs Rest Classification")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    print("Fitting Data...")
    trt_strt = time.time()
    svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma='auto', C=1000000)).fit(x_train, y_train)
    trt_end = time.time()
    print("SVM OneVsRest Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = svm_model_linear_ovr.predict(x_test)
    tst_end = time.time()
    print("SVM OneVsRest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = svm_model_linear_ovr.score(x_test, y_test)
    print("SVM OneVsRest Accuracy", str(accuracy * 100) + ' %')
    print("SVM OneVsRest MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))


def svm_one_vs_one_class(x_tr, x_ts, y_tr, y_ts):
    print("\nSVM One vs One Classification")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    print("Fitting Data...")
    trt_strt = time.time()
    svm_model_linear_ovo = SVC(kernel='rbf', gamma='auto', C=1000000).fit(x_train, y_train)
    trt_end = time.time()
    print("SVM OneVsOne Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = svm_model_linear_ovo.predict(x_test)
    tst_end = time.time()
    print("SVM OneVsRest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = svm_model_linear_ovo.score(x_test, y_test)
    print("SVM OneVsOne Accuracy", str(accuracy * 100) + ' %')
    print("SVM OneVsOne MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))


def try_all(x_tr, x_ts, y_tr, y_ts):
    print("\nClassification")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts
    C = 1000000  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
    lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.4, C=C).fit(x_train, y_train)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)
    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        predictions = clf.predict(x_test)
        accuracy = np.mean(predictions == y_test)
        print(accuracy)


def decision_tree_hype(model, x_train, y_train):
    parameters = {'max_depth': [1, 5, 10, 100, 1000]}
    tun = GridSearchCV(model, parameters, cv=5)
    tun.fit(x_train, y_train)
    return tun.best_params_


def knn_hype(model, x_train, y_train):
    parameters = {'n_neighbors': [1, 5, 10, 15]}
    tun = GridSearchCV(model, parameters, cv=5)
    tun.fit(x_train, y_train)
    return tun.best_params_


def random_forest_hype(model, x_train, y_train):
    parameters = {'n_estimators': [10, 150, 1000, 1500]}
    tun = GridSearchCV(model, parameters, cv=5)
    tun.fit(x_train, y_train)
    return tun.best_params_


def decision_tree(x_tr, x_ts, y_tr, y_ts):
    print("\nDecision Tree")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    tree = DecisionTreeClassifier(criterion='entropy', random_state=100, min_samples_leaf=100)

    print("Calculating Best HyperParameter...")
    best_params = decision_tree_hype(tree, x_train, y_train)
    maxdepth = best_params['max_depth']

    print("Fitting Data...")
    tree = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=maxdepth, min_samples_leaf=100)
    trt_strt = time.time()
    tree.fit(x_train, y_train)
    trt_end = time.time()
    print("Decision Tree Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = tree.predict(x_test)
    tst_end = time.time()
    print("Decision Tree Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    print("Decision Tree Accuracy", str(tree.score(x_test, y_test) * 100) + ' %')
    print("Decision Tree MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))


def knn(x_tr, x_ts, y_tr, y_ts):
    print("\nK-NN")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    k_nn = KNeighborsClassifier()

    print("Calculating Best HyperParameter...")
    best_params = knn_hype(k_nn, x_train, y_train)
    k = best_params['n_neighbors']

    print("Fitting Data...")
    trt_strt = time.time()
    k_nn = KNeighborsClassifier(n_neighbors=k)
    k_nn.fit(x_train, y_train)
    trt_end = time.time()
    print("K-NN Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = k_nn.predict(x_test)
    tst_end = time.time()
    print("K-NN Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    print("K-NN Accuracy", str(k_nn.score(x_test, y_test) * 100) + ' %')
    print("K-NN MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))


def random_forest(x_tr, x_ts, y_tr, y_ts):
    print("\nRandom Forest")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    rnd_frst = RandomForestClassifier(criterion='entropy', random_state=100, max_depth=10, min_samples_leaf=100)

    print("Calculating Best HyperParameter...")
    best_params = random_forest_hype(rnd_frst, x_train, y_train)
    estms = best_params['n_estimators']

    print("Fitting Data...")
    trt_strt = time.time()
    rnd_frst = RandomForestClassifier(n_estimators=estms, criterion='entropy', random_state=100, max_depth=10, min_samples_leaf=100)
    rnd_frst.fit(x_train, y_train)
    trt_end = time.time()
    print("Random Forest Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = rnd_frst.predict(x_test)
    tst_end = time.time()
    print("Random Forest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    print("Random Forest Accuracy", str(rnd_frst.score(x_test, y_test) * 100) + ' %')
    print("Random Forest MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
