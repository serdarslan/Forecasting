import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from lightfm.evaluation import auc_score
from lightfm import LightFM
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#add these deafults to init
BASE_PATH = '../input/'
BRAND_WEIGHT = 1
CATEGORY_WEIGHT = 1


class LigthFMModule:
    """
    lightfm model
    ---------------
    model

    data frames
    ---------------
    customer_df
    product_df
    order_df
    brand_df
    category_df
    department_df (??)

    entity lists
    ---------------
    customer_list
    product_list
    feature_list  --includes brand, category and department

    interaction matrixes
    ---------------
    customer_to_product_interaction_matrix
    product_to_feature_interaction_matrix

    recommendation result sets
    ---------------
    scores
    top_products
    """
    def __init__(self):
        self.customer_df = pd.read_csv(BASE_PATH + "customer.csv")
        self.product_df = pd.read_csv(BASE_PATH + "product.csv")
        self.order_df = pd.read_csv(BASE_PATH + "order.csv")
        self.brand_df = pd.read_csv(BASE_PATH + "brand.csv")
        self.category_df = pd.read_csv(BASE_PATH + "category.csv")

    def prep_data(self):
        self.customer_list = np.sort(self.customer_df["customer_id"].unique())
        self.product_list = np.sort(self.product_df["product_id"].unique())
        brand = self.brand_df["brand_name"]
        category = self.category_df["category_name"]
        self.feature_list = pd.concat([brand, category], ignore_index=True).unique()

    def set_model(self, loss_function, learning_rate=0.2):
        self.model = LightFM(loss=loss_function, learning_rate=learning_rate)

    def fit_hybrid_model(self, epochs, num_threads, verbose=False):
        print("Using customer to product matrix and product features")
        self.model.fit(self.customer_to_product_interaction_matrix,
                                user_features=None,
                                item_features=self.product_to_feature_interaction_matrix,
                                sample_weight=None,
                                epochs=epochs,
                                num_threads=num_threads,
                                verbose=verbose)


    def fit_collaborative_model(self, epochs, num_threads, verbose=False):
        print("Using only customer to product matrix")
        self.model.fit(self.customer_to_product_interaction_matrix,
                       user_features=None,
                       item_features=None,
                       sample_weight=None,
                       epochs=epochs,
                       num_threads=num_threads,
                       verbose=verbose)

    def predict_hybrid_model(self, user_ids):
        scores = self.model.predict(user_ids=user_ids,
                                    item_ids=np.arange(self.customer_to_product_interaction_matrix.shape[1]))
        return scores

    def predict_collaborative_model(self, user_ids):
        scores = self.model.predict(user_ids=user_ids,
                                    item_ids=np.arange(self.customer_to_product_interaction_matrix.shape[1]))
        return scores

    def top_products(self, scores):
        tops = self.products[np.argsort(-scores)]
        return tops

    def create_customer_product_interaction_matrix(self):
        row = self.order_df["customer_id"]
        col = self.order_df["product_id"]
        value = self.order_df["count"]
        self.customer_to_product_interaction_matrix = coo_matrix((value, (row, col)), shape=(len(self.customer_list), len(self.product_list)))

    def create_product_feature_interaction_matrix(self):
        product_feature_df = self.product_df.merge(self.brand_df).merge(self.category_df)[["product_name", "brand_name", "category_name"]]
        product_feature_df["product_name"] = product_feature_df["product_name"]
        product_feature_df["brand_name"] = product_feature_df["brand_name"]
        product_feature_df["category_name"] = product_feature_df["category_name"]
        product_brand_df = product_feature_df[["product_name", "brand_name"]].rename(columns={"brand_name": "feature"})
        product_brand_df["feature_count"] = BRAND_WEIGHT
        product_category_df = product_feature_df[["product_name", "category_name"]].rename(
            columns={"category_name", "feature"})
        product_category_df["feature_count"] = CATEGORY_WEIGHT
        product_feature_df = pd.concat([product_brand_df, product_category_df], ignore_index=True)
        del product_feature_df
        del product_brand_df
        del product_category_df
        product_feature_df = product_feature_df.groupby(["product_name", "feature"], as_index=False)[
            "feature_count"].sum()
        row = product_feature_df["product_name"]
        col = product_feature_df["feature"]
        value = product_feature_df["feature_count"]
        self.product_to_feature_interaction_matrix = coo_matrix( value, (row, col),
                                                                 shape=(len(self.product_list), len(self.product_list)))

