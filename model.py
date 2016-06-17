import preprocess
from sklearn.metrics import mean_squared_error, make_scorer
from time import time


seed = 53

def performance_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5


def evaluate_model(clf, x_train, y_train, x_test, y_test):
    print '----------------------'
    print clf

    start = time()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print "RMSE: {}".format(performance_metric(y_test, y_pred))

    end = time() 
    print "Trained model in {:.4f} seconds".format(end - start) 



def basic_training(clf, x_train, x_test, y_train, y_test):
  
    print 'Training basic models... '

    evaluate_model(clf, x_train, y_train, x_test, y_test)

    print 'Done!'  

def cal_baseline(x_train, x_test, y_train, y_test):
    # Baseline 1: use mean
    print '----------------------'
    print 'Baseline: mean sales'
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=x_test[:, 5]))  #col 5 for mean

    # Baseline 2: use median
    print '----------------------'
    print 'Baseline: median sales'
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=x_test[:, 6]))  #col 6 for median

def tuned_training(clf, x_train, x_test, y_train, y_test):
    from sklearn.cross_validation import KFold
    from sklearn.grid_search import GridSearchCV
    
    start = time()
    print 'Tuning model...'

    params = {'max_features':['auto', 'sqrt'],
              'min_samples_leaf':[1, 10, 50],
              'n_estimators':[5,10,20]}

    # cross validation
    cv_sets = KFold(x_train.shape[0], 3, True, seed)
    score_func = make_scorer(performance_metric, greater_is_better=False)
    print "Grid search..."
    grid = GridSearchCV(clf, params, cv=cv_sets, scoring=score_func)
    grid = grid.fit(x_train, y_train)
    clf_opt = grid.best_estimator_  

    # predict with optimal model after fitting the data
    y_pred = clf_opt.predict(x_test)
    print "RMSE (tuned model): {}".format(performance_metric(y_test, y_pred))
    end = time() 
    print "Trained model in {:.4f} seconds".format(end - start) 

def main():  
    x_all, y_all = preprocess.build_feature_label()


    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, 
                                         test_size=0.3, random_state=seed)

    #cal_baseline(x_train, x_test, y_train, y_test)

    # Learning models
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(random_state=seed)

    #basic_training(clf, x_train, x_test, y_train, y_test)
  
    tuned_training(clf, x_train, x_test, y_train, y_test)




if __name__ == "__main__":
    main()

    


