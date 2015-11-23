"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in `centroids` that is closest to `location`. If two
    centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    return min (centroids, key=lambda x: distance(x, location))


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in `centroids`. Each item in
    `restaurants` should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    lst=[]
    for restaurant in restaurants:
        closest_centroid=find_closest(restaurant_location(restaurant), centroids)
        lst+=[[closest_centroid,restaurant]]
    return(group_by_first(lst))

def find_centroid(cluster):
    """Return the centroid of the `cluster` based on the locations of the
    restaurants."""
    lattitude=[]
    longitude=[]
    for elem in cluster:
        coordinates=restaurant_location(elem)
        lattitude+=[coordinates[0]]
        longitude+=[coordinates[1]]
    return [mean(lattitude), mean(longitude)]

def k_means(restaurants, k, max_updates=100):
    """Use k-means to group `restaurants` by location into `k` clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        clusters_lst=group_by_centroid(restaurants, old_centroids)
        centroids=[]
        for cluster in clusters_lst:
            centroids+=[find_centroid(cluster)]
        n += 1
    return centroids


def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for `user` by performing least-squares linear regression using `feature_fn`
    on the items in `restaurants`. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    S_xx, S_yy, S_xy= 0, 0, 0
    for elem in xs:
        S_xx+=(elem- mean(xs))**2
    for elem in ys:
        S_yy+=(elem- mean(ys))**2
    for x,y in zip(xs,ys):
        S_xy+=(x-mean(xs))*(y-mean(ys))
    b=S_xy/S_xx
    a=mean(ys)-b*mean(xs)
    r_squared=(S_xy**2)/(S_xx*S_yy)

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within `feature_fns` that gives the highest R^2 value
    for predicting ratings by the `user`; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    r_best, best_func=0, 0
    for feature_fn in feature_fns:
        predictor, r_squared=find_predictor(user, reviewed, feature_fn)
        if r_squared>r_best:
            r_best, best_func=r_squared, feature_fn
    predict,r_square= find_predictor(user, reviewed, best_func)
    return predict



def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of `restaurants` by `user` using the best
    predictor based a function from `feature_fns`.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    dictionary={}
    for restaurant in restaurants:
        name=restaurant_name(restaurant)
        if restaurant in reviewed:
            dictionary[name]=user_rating(user,name)
        else:
            dictionary[name]=predictor(restaurant)
    return dictionary


def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_rating,
            restaurant_price,
            restaurant_num_ratings,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
