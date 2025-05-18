from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_synthetic(polynomial=False, n_samples=100, n_features=10, noise=0.1, random_state=42):
    x, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)

    if polynomial:
        poly = PolynomialFeatures(degree=3, include_bias=False)
        x = poly.fit_transform(x)

    return x, y


def load_california(scale=False, two_dim=True):
    data = fetch_california_housing(as_frame=False)
    x, y = data.data, data.target

    if scale:
        x = StandardScaler().fit_transform(x)

    if two_dim:
        return x, y

    return data
