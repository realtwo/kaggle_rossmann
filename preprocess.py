# -*- coding: utf-8 -*-

import pandas as pd
 

def cal_per_store_sales_summary(df_sales):
    
    # Only use data when store is open
    df_sales = df_sales[df_sales['Open']==1]
    sale_mean = df_sales[['Store','Sales'] ].groupby('Store').mean()
    sale_median = df_sales[['Store','Sales'] ].groupby('Store').median()
    

    sale_mean.columns= ['MeanSales']   
    sale_median.columns= ['MedianSales']

    print 'Stats on sales'
    print sale_mean.describe()
    print sale_median.describe()

    df_sales = df_sales.join(sale_mean, on='Store')
    df_sales = df_sales.join(sale_median, on='Store')

    return df_sales
    



def map_txt_to_num(s):
    d = {} #dict to map char to int
    for i, v in enumerate(s.unique().tolist()):
        d[v] = i
    return s.map(d)


def build_feature_label():  
    df_sales = pd.read_csv('train.csv', header=0)
    df_store = pd.read_csv('store.csv', header=0)

    # Data exploration
    print 'Columns in sales data: {}'.format(df_sales.columns.values)
    print 'Columns in store data: {}'.format(df_store.columns.values)

    print 'Total entries in sales: {}'.format(df_sales.shape[0])
    print 'Total entries in store: {}'.format(df_store.shape[0])
    
#    print df_sales.Store.unique()
#    print df_sales.DayOfWeek.unique()
    print df_sales.Sales.describe()
    print df_sales.Customers.describe()



    # calculate per store mean and median sales and add to sales
    df_sales= cal_per_store_sales_summary(df_sales)

    # Merge the sales data with store info
    df = pd.merge(df_sales, df_store, on='Store') #store info


    # Only interested in open stores
    df = df[df['Open']==1]

    col_total = df.shape[1]
    row_total = df.shape[0]

    label = 'Sales'
    features = df.columns.tolist()
    features.remove(label)

    feat_to_drop = [] # list of features to drop later

    print "Merged data has {} columns and {} rows".format(col_total, row_total)

    print df.info()
    

    # Drop columns that have too many missing data: automatic
    for v in features: 
        if df[v].count()<0.7*row_total:
            feat_to_drop.append(v)
            print "Feature {} has too many missing values. Drop it.".format(v)
    print "Features to drop due to too many missing values: {}".format(feat_to_drop)
    

    # Fill rest of the missing values
    df.fillna(-99, inplace=True)

    # Convert text feature to number
    df['StoreType'] = map_txt_to_num(df['StoreType'])
    df['Assortment'] = map_txt_to_num(df['Assortment'])
    df['StateHoliday'] = map_txt_to_num(df['StateHoliday'])

    # Drop other unnecessary features 
    feat_to_drop.append('Open') # Drop, as all stores are open
    feat_to_drop.append('Date')
    feat_to_drop.append('Store')
    features = [v for v in features if v not in feat_to_drop]

    # Final data

    print 'Features: {}'.format(features)

    x_all = df[features]
    y_all = df[label]

    print x_all.info()

    # Visualization
    #import matplotlib.pyplot as plt
    #plt.plot(x_all.Customers, y_all)
    #plt.show()

    return x_all, y_all



if __name__ == "__main__":

    x_all, y_all = build_feature_label()

    


