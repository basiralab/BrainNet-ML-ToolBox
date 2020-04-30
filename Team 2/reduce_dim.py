# Code Owners: Göktuğ Güvercin - Uğur Tepecik - Ege Apak
# Code Editor: Göktuğ Güvercin


from read_write import *


def create_dataframe(dataset):

    """
    * This function takes "dataset" array as an argument, and creates a data frame object "df".
      This dataframe object is needed to compute correlation coefficient matrix.
    * Finally, the method changes names of columns in data frame for easy use.

    Parameters
    ----------
    dataset: It is 2D numpy array. Its last column should contain target scores (labels).
    :return: data frame object
    """

    df = pd.DataFrame(dataset)
    df.columns = [i for i in range(len(df.columns) - 1)] + ["Labels"]
    return df


def find_pcc_features(df, nof_features):

    """
    * This method at first computes correlation matrix by using built-in corr() method.
    * Then, one of mutually-correlated features is eliminated. For example, feature A and feature B
      are well-correlated to each other. In this case, these features have similar behavior and
      similar effect on classification task, so we can discard one of them. We do not need to keep both of them.

    * After that, the row which stores the correlation between features and label is extracted.
      Absolute of that row is computed, because negative value only refers to inverse relation.
      We do not care forward or inverse relation; we care most related (max absolute values) correlations.
    * Then, correlation values are sorted in descending order.
    * Finally, indices of the features correlated to labels are stored in the list "pcc_features".

    * To summarize, MRMR (minimum redundancy maximum relevance) feature selection algorithm is performed.
      The features chosen by this algorithm is called pearson-correlation-coefficient (pcc) features.

    Parameters
    ----------
    df: data frame object
    nof_features: the number of features that you reduce the dimension to
    :return: a list of indices of the features which are highly-correlated to labels
    """

    pcc_features = []
    corr_features = []
    corr_matrix = df.corr()

    # determining similar (mutually-correlated) features
    for i in range(len(corr_matrix.columns) - 1):
        for j in range(i):
            if np.abs(corr_matrix[i][j]) > 0.75:
                corr_features.append(i)

    corr_matrix = corr_matrix.drop(corr_features)  # eliminating similar features
    corr_label = corr_matrix["Labels"].abs()  # taking absolute of correlations

    sorted_corr_label = corr_label.sort_values(na_position="last", ascending=False)
    feature_names = sorted_corr_label.index

    for i in range(1, nof_features + 1): # taking most informative n features
        pcc_features.append(feature_names[i])

    return pcc_features


def pcc_transform(features, indices):

    """
    This method takes transpose of "features" array to access the features easily.
    Our dataset is in the dimension 120 x 595, which means that each row refers to one sample.
    I want to keep most correlated features (indices), and remove the other features.
    To accomplish this, each feature must be represented a list.
    In that list, all values which that feature took across all samples must be stored.
    This is only possible by taking transpose of "features" array

    Parameters
    ----------
    features: two dimensional numpy array representing our samples without labels
    indices: index values of most correlated features
    :return:
    """

    features_T = features.T
    features_T = features_T[indices]
    features = features_T.T
    return features


def apply_MRMR(nof_features, tra_dataset, tst_features):

    """
    This method creates a data frame to be able to compute correlation matrix.
    Then, index values of most correlated n features are determined.
    By using these index values, training and testing features are reduced to lower dimension with pcc_transfrom().

    Parameters
    ----------
    nof_features: the number of features that you reduce the dimension to
    tra_dataset: Two dimensional numpy array (training set). Its last column refers to target scores (labels)
    tst_features: Two dimensional numpy array (testing set). It does not contain target scores (labels)
    :return: reduced training and testing set
    """

    df = create_dataframe(tra_dataset)
    pcc_indices = find_pcc_features(df, nof_features)

    tra_features = tra_dataset[:, :len(tra_dataset[0]) - 1]
    pcc_tra_features = pcc_transform(tra_features, pcc_indices)
    pcc_tst_features = pcc_transform(tst_features, pcc_indices)

    return pcc_tra_features, pcc_tst_features

