import pandas as pd
import numpy as np # numpy for sure
from scipy.sparse import coo_matrix # for constructing sparse matrix

#import lightfm
from lightfm.evaluation import auc_score
from lightfm import LightFM


import time

print("starting")
base_path = 'C:\\Users\\Serdar\\Documents\\pony\\data\\instacart\\'
aisles = pd.read_csv(base_path + "aisles.csv")
departments = pd.read_csv(base_path + "departments.csv")
orders = pd.read_csv(base_path + "orders.csv")
order_products__prior = pd.read_csv(base_path + "order_products__prior.csv")
order_products__train = pd.read_csv(base_path + "order_products__train.csv")
products = pd.read_csv(base_path + "products.csv")

# removing aisles with aisle == "missing" and aisle == "other"
# removing departments with department == "missing" and department == "other"

aisles = aisles[aisles["aisle"].apply(lambda x: x != "missing" and x != "other")]
departments = departments[departments["department"].apply(lambda x: x != "missing" and x != "other")]


def get_user_list(df, user_column):
    """

    creating a list of user from dataframe df, user_column is a column
    consisting of users in the dataframe df

    """

    return np.sort(df[user_column].unique())


def get_item_list(df, item_name_column):
    """

    creating a list of items from dataframe df, item_column is a column
    consisting of items in the dataframe df

    return to item_id_list and item_id2name_mapping

    """

    item_list = df[item_name_column].unique()

    return item_list


def get_feature_list(aisle_df, department_df, aisle_name_column, department_name_column):
    aisle = aisle_df[aisle_name_column]
    department = department_df[department_name_column]

    return pd.concat([aisle, department], ignore_index=True).unique()


# creating user_id, item_id, and features_id

def id_mappings(user_list, item_list, feature_list):
    """

    Create id mappings to convert user_id, item_id, and feature_id

    """
    user_to_index_mapping = {}
    index_to_user_mapping = {}
    for user_index, user_id in enumerate(user_list):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id

    item_to_index_mapping = {}
    index_to_item_mapping = {}
    for item_index, item_id in enumerate(item_list):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id

    feature_to_index_mapping = {}
    index_to_feature_mapping = {}
    for feature_index, feature_id in enumerate(feature_list):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id

    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping


def get_user_product_interaction(orders_df, order_products_train_df, order_products_test_df, products_df):
    # creating a dataframe consists of TWO columns user_id, and product_name (product bought by the user) for the train data
    user_to_product_train_df = orders_df[orders_df["eval_set"] == "prior"][["user_id", "order_id"]]. \
        merge(order_products_train_df[["order_id", "product_id"]]).merge(products_df[["product_id", "product_name"]]) \
        [["user_id", "product_name"]].copy()

    # giving rating as the number of product purchase count
    user_to_product_train_df["product_count"] = 1
    user_to_product_rating_train = user_to_product_train_df.groupby(["user_id", "product_name"], as_index=False)[
        "product_count"].sum()

    # creating a dataframe consists of TWO columns user_id, and product_name (product bought by the user) for the test data
    user_to_product_test_df = orders_df[orders_df["eval_set"] == "train"][["user_id", "order_id"]]. \
        merge(order_products_test_df[["order_id", "product_id"]]).merge(products_df[["product_id", "product_name"]]) \
        [["user_id", "product_name"]].copy()

    # giving rating as the number of product purchase count (including the previous purchase in the training data)
    user_to_product_test_df["product_count"] = 1
    user_to_product_rating_test = user_to_product_test_df.groupby(["user_id", "product_name"], as_index=False)[
        "product_count"].sum()

    # merging with the previous training user_to_product_rating_training

    user_to_product_rating_test = user_to_product_rating_test. \
        merge(user_to_product_rating_train.rename(columns={"product_count": "previous_product_count"}),
              how="left").fillna(0)
    user_to_product_rating_test["product_count"] = user_to_product_rating_test.apply(
        lambda x: x["previous_product_count"] + \
                  x["product_count"], axis=1)
    user_to_product_rating_test.drop(columns=["previous_product_count"], inplace=True)

    return user_to_product_rating_train, user_to_product_rating_test


def get_interaction_matrix(df, df_column_as_row, df_column_as_col, df_column_as_value, row_indexing_map,
                           col_indexing_map):
    row = df[df_column_as_row].apply(lambda x: row_indexing_map[x]).values
    col = df[df_column_as_col].apply(lambda x: col_indexing_map[x]).values
    value = df[df_column_as_value].values

    return coo_matrix((value, (row, col)), shape=(len(row_indexing_map), len(col_indexing_map)))


def get_product_feature_interaction(product_df, aisle_df, department_df, aisle_weight=1, department_weight=1):
    item_feature_df = product_df.merge(aisle_df).merge(department_df)[["product_name", "aisle", "department"]]

    # start indexing
    item_feature_df["product_name"] = item_feature_df["product_name"]
    item_feature_df["aisle"] = item_feature_df["aisle"]
    item_feature_df["department"] = item_feature_df["department"]

    # allocate aisle and department into one column as "feature"

    product_aisle_df = item_feature_df[["product_name", "aisle"]].rename(columns={"aisle": "feature"})
    product_aisle_df["feature_count"] = aisle_weight  # adding weight to aisle feature
    product_department_df = item_feature_df[["product_name", "department"]].rename(columns={"department": "feature"})
    product_department_df["feature_count"] = department_weight  # adding weight to department feature

    # combining aisle and department into one
    product_feature_df = pd.concat([product_aisle_df, product_department_df], ignore_index=True)

    # saving some memory
    del item_feature_df
    del product_aisle_df
    del product_department_df

    # grouping for summing over feature_count
    product_feature_df = product_feature_df.groupby(["product_name", "feature"], as_index=False)["feature_count"].sum()

    return product_feature_df

# create the user, item, feature lists
users = get_user_list(orders, "user_id")
items = get_item_list(products, "product_name")
features = get_feature_list(aisles, departments, "aisle", "department")

# generate mapping, LightFM library can't read other than (integer) index
user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping = id_mappings(users, items, features)

user_to_product_rating_train, user_to_product_rating_test = get_user_product_interaction(orders, order_products__prior,
                                                                                        order_products__train, products)
product_to_feature = get_product_feature_interaction(product_df = products,
                                                     aisle_df = aisles,
                                                     department_df = departments,
                                                     aisle_weight=1,
                                                     department_weight=1)
# generate user_item_interaction_matrix for train data
user_to_product_interaction_train = get_interaction_matrix(user_to_product_rating_train, "user_id",
                                                    "product_name", "product_count", user_to_index_mapping, item_to_index_mapping)

# generate user_item_interaction_matrix for test data
user_to_product_interaction_test = get_interaction_matrix(user_to_product_rating_test, "user_id",
                                                    "product_name", "product_count", user_to_index_mapping, item_to_index_mapping)

# generate item_to_feature interaction
product_to_feature_interaction = get_interaction_matrix(product_to_feature, "product_name", "feature",  "feature_count",
                                                        item_to_index_mapping, feature_to_index_mapping)

# initialising model with warp loss function
model_without_features = LightFM(loss = "warp")

# fitting into user to product interaction matrix only / pure collaborative filtering factor
start = time.time()
#===================

model_without_features.fit(user_to_product_interaction_train,
          user_features=None,
          item_features=None,
          sample_weight=None,
          epochs=1,
          num_threads=4,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# auc metric score (ranging from 0 to 1)

start = time.time()
#===================

auc_without_features = auc_score(model = model_without_features,
                        test_interactions = user_to_product_interaction_test,
                        num_threads = 4, check_intersections = False)
#===================
end = time.time()

print("time taken = {0:.{1}f} seconds".format(end - start, 2))
print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_without_features.mean(), 2))

# initialising model with warp loss function
model_with_features = LightFM(loss = "warp")

# fitting the model with hybrid collaborative filtering + content based (product + features)
start = time.time()
#===================


model_with_features.fit(user_to_product_interaction_train,
          user_features=None,
          item_features=product_to_feature_interaction,
          sample_weight=None,
          epochs=1,
          num_threads=4,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

start = time.time()
#===================
auc_with_features = auc_score(model = model_with_features,
                        test_interactions = user_to_product_interaction_test,
                        train_interactions = user_to_product_interaction_train,
                        item_features = product_to_feature_interaction,
                        num_threads = 4, check_intersections=False)
#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))


def combined_train_test(train, test):
    """

    test set is the more recent rating/number_of_order of users.
    train set is the previous rating/number_of_order of users.
    non-zero value in the test set will replace the elements in
    the train set matrices
    """
    # initialising train dict
    train_dict = {}
    for train_row, train_col, train_data in zip(train.row, train.col, train.data):
        train_dict[(train_row, train_col)] = train_data

    # replacing with the test set

    for test_row, test_col, test_data in zip(test.row, test.col, test.data):
        train_dict[(test_row, test_col)] = max(test_data, train_dict.get((test_row, test_col), 0))

    # converting to the row
    row_element = []
    col_element = []
    data_element = []
    for row, col in train_dict:
        row_element.append(row)
        col_element.append(col)
        data_element.append(train_dict[(row, col)])

    # converting to np array

    row_element = np.array(row_element)
    col_element = np.array(col_element)
    data_element = np.array(data_element)

    return coo_matrix((data_element, (row_element, col_element)), shape=(train.shape[0], train.shape[1]))

user_to_product_interaction = combined_train_test(user_to_product_interaction_train,
                                                 user_to_product_interaction_test)
# retraining the final model with combined dataset

final_model = LightFM(loss = "warp")

# fitting to combined dataset with pure collaborative filtering result

start = time.time()
#===================

final_model.fit(user_to_product_interaction,
          user_features=None,
          item_features=None,
          sample_weight=None,
          epochs=1,
          num_threads=4,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))


class recommendation_sampling:

    def __init__(self, model, items=items, user_to_product_interaction_matrix=user_to_product_interaction,
                 user2index_map=user_to_index_mapping):

        self.user_to_product_interaction_matrix = user_to_product_interaction_matrix
        self.model = model
        self.items = items
        self.user2index_map = user2index_map

    def recommendation_for_user(self, user):

        # getting the userindex

        userindex = self.user2index_map.get(user, None)

        if userindex == None:
            return None

        users = [userindex]

        # products already bought

        known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]

        # scores from model prediction
        scores = self.model.predict(user_ids=users,
                                    item_ids=np.arange(self.user_to_product_interaction_matrix.shape[1]))

        # top items

        top_items = self.items[np.argsort(-scores)]

        # printing out the result
        print("User %s" % user)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("                  %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("                  %s" % x)

recom = recommendation_sampling(model = final_model)
recom.recommendation_for_user(2)
recom.recommendation_for_user(10)
