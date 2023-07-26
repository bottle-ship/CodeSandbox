import typing as t

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import bernoulli


def make_user_retention_dataset(n: int = 10000, random_state: t.Optional[int] = None):
    """ The generative model for our subscriber retention example."""

    rng = np.random.default_rng(seed=random_state)
    data = pd.DataFrame()

    # the number of sales calls made to this customer
    data["Sales calls"] = rng.uniform(0, 4, size=(n,)).round()

    # the number of sales calls made to this customer
    data["Interactions"] = data["Sales calls"] + rng.poisson(0.2, size=(n,))

    # the health of the regional economy this customer is a part of
    data["Economy"] = rng.uniform(0, 1, size=(n,))

    # the time since the last product upgrade when this customer came up for renewal
    data["Last upgrade"] = rng.uniform(0, 20, size=(n,))

    # how much the user perceives that they need the product
    data["Product need"] = (data["Sales calls"] * 0.1 + rng.normal(0, 1, size=(n,)))

    # the fractional discount offered to this customer upon renewal
    data["Discount"] = ((1 - expit(data["Product need"])) * 0.5 + 0.5 * rng.uniform(0, 1, size=(n,))) / 2

    # What percent of the days in the last period was the user actively using the product
    data["Monthly usage"] = expit(data["Product need"] * 0.3 + rng.normal(0, 1, size=(n,)))

    # how much ad money we spent per user targeted at this user (or a group this user is in)
    data["Ad spend"] = data["Monthly usage"] * rng.uniform(0.9, 0.99, size=(n,))
    data["Ad spend"] = data["Ad spend"] + (data["Last upgrade"] < 1) + (data["Last upgrade"] < 2)

    # how many bugs did this user encounter in the since their last renewal
    data["Bugs faced"] = np.array([rng.poisson(v * 2) for v in data["Monthly usage"]])

    # how many bugs did the user report?
    data["Bugs reported"] = (data["Bugs faced"] * expit(data["Product need"])).round()

    # did the user renew?
    data["Did renew"] = expit(
        7 * (
                0.18 * data["Product need"]
                + 0.08 * data["Monthly usage"]
                + 0.1 * data["Economy"]
                + 0.05 * data["Discount"]
                + 0.05 * rng.normal(0, 1, size=(n,))
                + 0.05 * (1 - data['Bugs faced'] / 20)
                + 0.005 * data["Sales calls"]
                + 0.015 * data["Interactions"]
                + 0.1 / (data["Last upgrade"]/4 + 0.25)
                + data["Ad spend"] * 0.0 - 0.45
        )
    )

    # in real life we would make a random draw to get either 0 or 1 for if the
    # customer did or did not renew. but here we leave the label as the probability
    # so that we can get less noise in our plots. Uncomment this line to get
    # noiser causal effect lines but the same basic results
    data["Did renew"] = bernoulli.rvs(data["Did renew"])

    x = data.drop(columns=["Did renew", "Product need", "Bugs faced"])
    y = data.reindex(columns=["Did renew"])
    x_unmeasured = data.reindex(columns=["Product need", "Bugs faced"])

    return (x, y), x_unmeasured
