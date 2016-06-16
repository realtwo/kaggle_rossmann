import preprocess
from sklearn.metrics import mean_squared_error, make_scorer
from time import time

from sklearn.cross_validation import cross_val_score

seed = 53

def performance_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5


def fit_and_predict(clf, x_train, y_train, x_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def evaluate_model(clf, x_train, y_train, x_test, y_test):
    print '----------------------'
    print clf

    start = time()
    
    y_pred = fit_and_predict(clf, x_train, y_train, x_test)
    print "RMSE: {}".format(performance_metric(y_test, y_pred))

    end = time() 
    print "Trained model in {:.4f} seconds".format(end - start) 



def basic_training(x_train, x_test, y_train, y_test):
  
    print 'Training basic models... '

    # Learning models
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(random_state=seed)
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

def tuning():
    # cross validation
    #score_func = make_scorer(performance_metric, greater_is_better=False)
    #print "Cross validating..."
    #scores = cross_val_score(clf, x_train, y_train, cv=3, scoring=score_func)
    #print "CV result: RMSE: {}".format(scores)

    pass




def main():  
    x_all, y_all = preprocess.build_feature_label()


    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, 
                                         test_size=0.3, random_state=seed)

    cal_baseline(x_train, x_test, y_train, y_test)


    basic_training(x_train, x_test, y_train, y_test)
  


if __name__ == "__main__":
    main()

    


