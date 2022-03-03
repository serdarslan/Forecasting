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

# creating a dataframe consists of TWO columns user_id, and product_name (product bought by the user) for the train data
user_to_product_train_df = orders[orders["eval_set"] == "prior"][["user_id", "order_id"]]. \
    merge(order_products__train[["order_id", "product_id"]]).merge(products[["product_id", "product_name"]]) \
    [["user_id", "product_name"]].copy()

# giving rating as the number of product purchase count
user_to_product_train_df["product_count"] = 1
user_to_product_rating_train = user_to_product_train_df.groupby(["user_id", "product_name"], as_index=False)[
    "product_count"].sum()

print("finish")