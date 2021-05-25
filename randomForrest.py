from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
from matplotlib import pyplot as plt
import numpy as np
import math


def rf(xs, y, n_estimators=40, max_samples=None,
       max_features='sqrt', min_samples_leaf=5, **kwargs):
    return RandomForestClassifier(criterion='entropy', n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)

#def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)

def r_mse(pred, y): return np.sum(pred == y)/len(y)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

if __name__ == "__main__":
    AllIn_test =     np.array(loadmat("Data/ALL_AllIn_test.mat")['AllIn_test'])
    AllIn_train =    np.array(loadmat("Data/ALL_AllIn_train.mat")['AllIn_train'])
    Disjunct_train = np.array(loadmat("Data/ALL_Disjunct_train.mat")['Disjunct_train'])
    Disjunct_test =  np.array(loadmat("Data/ALL_Disjunct_test.mat")['Disjunct_test'])

    lab_train_All = AllIn_train[:,0]
    data_train_All = AllIn_train[:,1:]
    lab_test_All = AllIn_test[:,0]
    data_test_All = AllIn_test[:,1:]
    lab_train_Dis = Disjunct_train[:,0]
    data_train_Dis =Disjunct_train[:,1:]
    lab_test_Dis = Disjunct_test[:,0]
    data_test_Dis =Disjunct_test[:,1:]

    model_all = rf(data_train_All, lab_train_All)
    model_dis = rf(data_train_Dis, lab_train_Dis)

    print("All in:\t validation accuracy: {:.3f}  test accuracy: {:.3f}".format(m_rmse(model_all, data_train_All, lab_train_All), m_rmse(model_all, data_test_All, lab_test_All)))
    print("DisJunct:\t validation accuracy: {:.3f}  test accuracy: {:.3f}".format(m_rmse(model_dis, data_train_Dis, lab_train_Dis), m_rmse(model_dis, data_test_Dis, lab_test_Dis)))
    print(np.array(model_all.feature_importances_).shape)
    featureIMG = np.reshape(np.array(model_all.feature_importances_), [18,18])

    fig, ax = plt.subplots()

    im, cbar = heatmap(featureIMG, range(18), range(18), ax=ax, cmap="YlGn", cbarlabel="Feature Importance")

    #plt.imshow(featureIMG, interpolation='nearest')
    fig.tight_layout()
    plt.show()

    preds = np.stack([t.predict(data_test_All) for t in model_all.estimators_])
    plt.plot([r_mse(preds[:i+1].mean(0), lab_test_All) for i in range(40)])
    plt.show()
    #plt.plot(preds)
