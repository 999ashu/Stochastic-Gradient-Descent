from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_california(scale=False, two_dim=True):
    data = fetch_california_housing(as_frame=False)
    x, y = data.data, data.target

    if scale:
        x = StandardScaler().fit_transform(x)

    if two_dim:
        return x, y

    return data
