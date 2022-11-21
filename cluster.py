def cluster_analysis_2d(df, y_arg):
########################################################
# y_arg should be non-numeric data (for current version)
# Will not use non-numeric data as dimensions
#
# Runtime gets long after ~50 variables
########################################################    


    # import dependencies
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score


    # prepare data
    encode = LabelEncoder()
    target_labels = encode.fit_transform(df[y_arg])

    data_all = df[df.columns.drop(y_arg)]
    data_all = data_all.select_dtypes(include=['int64','float64'])




    # set up pipeline
    n_clusters = len(df[y_arg].unique())

    preprocessor = Pipeline([('scaler',MinMaxScaler())])
    clusterer = Pipeline([('kmeans', KMeans(n_clusters = n_clusters, init = 'k-means++', max_iter = 500, random_state=34))])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clusterer', clusterer)
        ])

    # loop through all 2d combinations of dimensions
    d1 = data_all.columns; d2 = data_all.columns
    max_rand_score = -1
    sil_score_best = 0; dimension1_best = ''; dimension2_best = ''

    for dimension1 in d1:
        d2 = d2.drop(dimension1)
        for dimension2 in d2:
            
            # prepare data
            data = np.array(data_all[[dimension1,dimension2]]) 

            # fit data
            pipe.fit(data)


            # performance metrics
            clustered_data = pipe['preprocessor'].transform(data)
            predicted_labels = pipe['clusterer']['kmeans'].labels_

            sil_score = silhouette_score(clustered_data,predicted_labels)
            rand_score = adjusted_rand_score(target_labels,predicted_labels)

            if (rand_score > max_rand_score):
                max_rand_score = rand_score
                sil_score_best = sil_score
                dimension1_best = dimension1
                dimension2_best = dimension2


    # plot results
    data_best = np.array(data_all[[dimension1_best,dimension2_best]])
    pipe.fit(data_best)
    clustered_data = pipe['preprocessor'].transform(data_best)
    predicted_labels = pipe['clusterer']['kmeans'].labels_

    to_plot = pd.DataFrame(clustered_data,columns = [dimension1_best,dimension2_best])

    to_plot['Cluster'] = predicted_labels
    to_plot[y_arg] = encode.inverse_transform(target_labels)

    plt.style.use('fivethirtyeight')
    plt.figure()
    scatter = sns.scatterplot(
        dimension1_best, dimension2_best, data=to_plot, hue='Cluster', style=y_arg, s=100, palette='colorblind'
    )
    scatter.set_title(f'Clustering results for {y_arg}')

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()
    print(f'silhouette score is: {sil_score_best}, rand score: {max_rand_score}')