import pandas as pd


def add_tenure_bucket(data):
    data = data.copy()
    data["TenureBucket"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype(str)
    return data