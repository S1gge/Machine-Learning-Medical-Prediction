import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier


class Disease_Prediction:

    def __init__(self, df, feature):
        self.df = df
        self.X = self.df.drop(feature, axis=1)
        self.y = self.df[feature]

    def split_data(self, test_size=0.3, val_size=0.5, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=val_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def scaling(self, X_train, X_val, X_test):
        standard = StandardScaler()
        normalize = MinMaxScaler()
        X_train_stan = standard.fit_transform(X_train)
        X_val_stan = standard.transform(X_val)
        X_test_stan = standard.transform(X_test)

        X_train_scale = normalize.fit_transform(X_train_stan)
        X_val_scale = normalize.transform(X_val_stan)
        X_test_scale = normalize.transform(X_test_stan)
        return X_train_scale, X_val_scale, X_test_scale
    

    def grid_search(self, estimator, X_train_scale, y_train, param_grid, cv=5 ):
        classifier = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring="recall")
        classifier.fit(X_train_scale, y_train)
        print(classifier.best_estimator_.get_params())
        return classifier


    def evaluate(self, classifier, X_val_scale, y_val):
        y_pred_val = classifier.predict(X_val_scale)
        print(classification_report(y_val, y_pred_val))
        cm = confusion_matrix(y_val, y_pred_val)
        ConfusionMatrixDisplay(cm).plot()
        return


    def log_reg(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)
        
        param_grid = [
            {"l1_ratio": np.linspace(0, 1, 20),
             'penalty':['elasticnet'],
             'C' : [0.01, 0.1, 1, 10, 100],
             'solver': ['saga'],
             'max_iter': [1000]}
             ]

        classifier = self.grid_search(LogisticRegression(), X_train_scale, y_train, param_grid, cv=5)
        self.evaluate(classifier, X_val_scale, y_val)

        log_reg_best_model = classifier.best_estimator_
        return log_reg_best_model


    def knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)
        param_grid = {"n_neighbors": list(range(1,50))}
        
        classifier = self.grid_search(KNeighborsClassifier(),X_train_scale, y_train, param_grid, cv=5)
        self.evaluate(classifier, X_val_scale, y_val)

        knn_best_model = classifier.best_estimator_
        return knn_best_model
    
    def random_forest(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)
        
        param_grid = [
            {"n_estimators": [200], 
             "criterion": ["gini"],
             "max_features":["log2"]}
            ]
        
        classifier = self.grid_search(RandomForestClassifier(), X_train_scale, y_train, param_grid, cv=5)
        self.evaluate(classifier, X_val_scale, y_val)

        random_forest_best_model = classifier.best_estimator_
        return random_forest_best_model
    
    def voting_clf(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)

        log_reg = self.log_reg()
        knn = self.knn()
        random_forest = self.random_forest()
        vote_clf = VotingClassifier(
            [('log_reg',log_reg),
             ('knn', knn),
             ('rf', random_forest)],
             voting='soft'
        )
        vote_clf.fit(X_train_scale, y_train)
        self.evaluate(vote_clf, X_val_scale, y_val)
        return vote_clf