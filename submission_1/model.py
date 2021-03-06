import preprocess
from sklearn.metrics import mean_squared_error, make_scorer
from time import time
from sklearn.cross_validation import train_test_split


seed = 53

def performance_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5

def basic_training(clf, x_train, x_test, y_train, y_test):
  
    print '----------------------'
    print 'Basic training'
    print clf

    start = time()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print "RMSE: {}".format(performance_metric(y_test, y_pred))

    end = time() 
    print "Trained model in {:.4f} seconds".format(end - start) 

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


def grid_search(clf, x_train, y_train, params):
    from sklearn.cross_validation import KFold
    from sklearn.grid_search import GridSearchCV

    # cross validation
    cv_sets = KFold(x_train.shape[0], 3, True, seed)
    score_func = make_scorer(performance_metric, greater_is_better=False)
    print "Grid search..."
    grid = GridSearchCV(clf, params, cv=cv_sets, scoring=score_func)
    grid = grid.fit(x_train, y_train)
    return grid.best_estimator_      


def tuned_training(clf, x_train, x_test, y_train, y_test):
    print '---------------------'
    start = time()
    print 'Tuning model...'
    
    p1 = 'max_depth'
    params = {  p1: [16,20,24] }

    clf_opt = grid_search(clf, x_train, y_train, params)

    print "Parameter {} is {} for the optimal model.".format(p1, clf_opt.get_params()[p1])   

    # predict with optimal model after fitting the data
    y_pred = clf_opt.predict(x_test)
    rmse = performance_metric(y_test, y_pred)
    print "RMSE (tuned model): {}".format(rmse)
    end = time() 
    print "Trained model in {:.4f} seconds".format(end - start) 
    return rmse


def cal_sensitivity(clf, x_all, y_all, train_size):

    print '----------------------------'
    print 'Sensitivity analysis'
    rmse = []
    test_data = x_all[:6]
    test_result = y_all[:6]

    print 'Feature for price prediction:'
    print test_data
    print 'True sales price:'
    print test_result

    x_all = x_all[6:]
    y_all = y_all[6:]

    p1 = 'max_depth'
    params = { p1: [16,20,24] }

    for v in xrange(0,10):
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, 
                                         train_size=train_size, random_state=v)
        clf_opt = grid_search(clf, x_train, y_train, params)
        y_pred = clf_opt.predict(test_data)
        print 'Sale prediction for run #{}: {}'.format(v, y_pred)
        
        y_pred = clf_opt.predict(x_test)
        rmse.append(performance_metric(y_test, y_pred))

    print 'RSME for 10 random splits: {}'.format(rmse)



def main():  
    x_all, y_all = preprocess.build_feature_label()


    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, 
                                         test_size=0.3, random_state=seed)

    train_size = x_train.shape[0]

    cal_baseline(x_train, x_test, y_train, y_test)

    # Learning models
    from sklearn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor(random_state=seed)


    basic_training(clf, x_train, x_test, y_train, y_test)
  
    tuned_training(clf, x_train, x_test, y_train, y_test)

    cal_sensitivity(clf, x_all, y_all, train_size)


if __name__ == "__main__":
    main()

    


