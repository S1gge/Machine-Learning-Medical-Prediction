import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
        return classifier


    def evaluate(self, classifier, X_val_scale, y_val, 
                 labels=["No", "Yes"], show_class_report = True, conf_matrix=True):
        y_pred_val = classifier.predict(X_val_scale)

        if show_class_report:
            print(classification_report(y_val, y_pred_val))
        cm = confusion_matrix(y_val, y_pred_val)
        if conf_matrix:
            ConfusionMatrixDisplay(cm, display_labels=labels).plot()


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

        print(f'*** Logistic Regression Report ***')
        classifier = self.grid_search(LogisticRegression(), X_train_scale, y_train, param_grid, cv=5)
        self.evaluate(classifier, X_val_scale, y_val, show_class_report = True, conf_matrix=False)

        self.log_reg_best_model = classifier.best_estimator_
        return self.log_reg_best_model


    def knn(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)
        param_grid = [{"n_neighbors": list(range(1,50)),
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan', 'minkowski']}
                      ]
        
        print(f'*** KNeighborgs Report ***')
        classifier = self.grid_search(KNeighborsClassifier(),X_train_scale, y_train, param_grid, cv=5)
        self.evaluate(classifier, X_val_scale, y_val, show_class_report = True, conf_matrix=False)

        self.knn_best_model = classifier.best_estimator_
        return self.knn_best_model
    
    def random_forest(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)
        
        param_grid = [
            {"n_estimators": [100, 150, 200, 300],
             "max_depth" : [None],
             "criterion": ["gini", "entropy"],
             "max_leaf_nodes": [6],
             "max_features":["sqrt", "log2"]}
            ]
        
        print(f'*** Random Forest report ***')
        classifier = self.grid_search(RandomForestClassifier(), X_train_scale, y_train, param_grid, cv=5)
        self.evaluate(classifier, X_val_scale, y_val, show_class_report = True, conf_matrix=False)

        self.random_forest_best_model = classifier.best_estimator_
        return self.random_forest_best_model
    
    def voting_clf(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)

        log_reg = self.log_reg_best_model
        knn = self.knn_best_model
        random_forest = self.random_forest_best_model

        vote_clf = VotingClassifier(
            [('log_reg',log_reg),
             ('knn', knn),
             ('rf', random_forest)],
             voting='hard'
        )
        vote_clf.fit(X_train_scale, y_train)
        self.evaluate(vote_clf, X_val_scale, y_val, show_class_report = True, conf_matrix=True)

        return vote_clf
    
    def test(self, model):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        X_train_scale, X_val_scale, X_test_scale = self.scaling(X_train, X_val, X_test)

        X_train=np.concatenate((X_train_scale,X_val_scale))
        y_train=np.concatenate((y_train,y_val))
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_scale)

        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot()