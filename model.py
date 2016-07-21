import preprocess
from sklearn.metrics import mean_squared_error, make_scorer
from time import time
from sklearn.cross_validation import train_test_split
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np

seed = 53

def performance_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5

def basic_training(clf, x_train, x_test, y_train, y_test, plot_importance=False):
  
    print '----------------------'
    print 'Basic training'
    print clf

    start = time()


    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print "RMSE: {}".format(performance_metric(y_test, y_pred))

    end = time() 
    print "Trained model in {:.4f} seconds".format(end - start) 

    # plot feature importance
    if plot_importance:
        importance = clf.feature_importances_
        importance = 100.0 * (importance / importance.max())

        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.figure()
        plt.barh(pos, importance[sorted_idx], align='center')
        plt.yticks(pos, x_train.columns[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()

    print 'Done basic training!'  

    return y_pred

def cal_baseline(x_test, y_test):

    # Baseline 1: use mean
    print '----------------------'
    print 'Baseline: mean sales'
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=x_test['MeanSales']))  

    # Baseline 2: use median
    print '----------------------'
    print 'Baseline: median sales'
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=x_test['MedianSales']))  


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

def evaluate_model(x_train, x_test, y_train, y_test, plot_complexity = False):
    max_depth = range(2,30)
    rmse_train = []
    rmse_test = []

    for d in max_depth:
        clf = tree.DecisionTreeRegressor(random_state=seed, max_depth = d)
        y_pred_train = basic_training(clf, x_train, x_train, y_train, y_train)  #y_train
        y_pred_test = basic_training(clf, x_train, x_test, y_train, y_test)     #y_test
        rmse_train.append(performance_metric(y_train, y_pred_train))
        rmse_test.append(performance_metric(y_test, y_pred_test))

    print rmse_train
    print rmse_test

    if plot_complexity:
        plt.figure()
        plt.plot(max_depth, rmse_train, label='Training')
        plt.plot(max_depth, rmse_test, label='Testing')
        plt.xlabel('max depth')
        plt.ylabel('RMSE')
        plt.legend(loc='lower left')
        plt.show()


def main():  
    x_all, y_all = preprocess.build_feature_label()
   
    train_size = int(x_all.shape[0] * 0.7)
    test_size = x_all.shape[0] - train_size

    x_train = x_all[0:train_size]
    x_test = x_all[train_size:]

    y_train = y_all[0:train_size]
    y_test = y_all[train_size:]


    #cal_baseline(x_test, y_test)

    # Learning models
    clf = tree.DecisionTreeRegressor(random_state=seed)

    #from sklearn.ensemble import RandomForestRegressor
    #clf = RandomForestRegressor(random_state=seed)

    #basic_training(clf, x_train, x_test, y_train, y_test, plot_importance=True)
  
    #tuned_training(clf, x_train, x_test, y_train, y_test)

    #cal_sensitivity(clf, x_all, y_all, train_size)

    evaluate_model(x_train, x_test, y_train, y_test, plot_complexity=True)

if __name__ == "__main__":
    main()

    


