import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class Disease_Prediction:

    def __init__(self, df, feature):
        self.df = df
        self.X = self.df.drop(feature, axis=1)
        self.y = self.df[feature]
        

    def log_reg(self, scaler):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

        pipe_log = Pipeline([("scaler", scaler), ("log", LogisticRegression(
        solver="saga", max_iter=1000, penalty="elasticnet"))])
        l1_ratio = np.linspace(0,1, 20)
        param_grid_log = {"log__l1_ratio": l1_ratio}
        classifier_log = GridSearchCV(estimator=pipe_log, param_grid=param_grid_log, cv=5, scoring="accuracy")
        classifier_log.fit(X_train, y_train)

        classifier_log.best_estimator_.get_params()

        y_pred = classifier_log.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()


    def KNN(self, scaler):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

        pipe_KNN = Pipeline([("scaler", scaler), ("knn", KNeighborsClassifier())])
        param_grid_KNN = {"knn__n_neighbors": list(range(1,50))}
        classifier_KNN = GridSearchCV(estimator=pipe_KNN, param_grid=param_grid_KNN, cv=5, scoring="accuracy")
        classifier_KNN.fit(X_train, y_train)

        classifier_KNN.best_estimator_.get_params()

        y_pred = classifier_KNN.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()