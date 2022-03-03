import pandas as pd
import numpy as np # numpy for sure
from scipy.sparse import coo_matrix # for constructing sparse matrix

# base_path = 'C:\\Users\\Serdar\\Documents\\pony\\data\\instacart\\'
#
# df = pd.read_csv(base_path + "products_small.csv")
#
# for i in range(0,100):
#     row2 = df["product_id"].apply(lambda x: i).values
#
# row = df["product_id"].apply(lambda x: 1).values
# col = df["aisle_id"].apply(lambda x: 61).values
# value = df["department_id"].values
#
# print(row)
# print(col)
# print(value)
#
# matrix = [(222, 34, 23),
#           (333, 31, 11),
#           (444, 16, 21),
#           (555, 32, 22),
#           (666, 33, 27),
#           (777, 35, 12)]
# df = pd.DataFrame(matrix, columns=list('abc'))
#
# print(df)
#
# df2 = df['a'].apply(lambda x: x*2).values
# index_mapping = {}
# index_mapping[222] = 0
# index_mapping[333] = 1
# index_mapping[444] = 2
# index_mapping[555] = 3
# index_mapping[666] = 4
# index_mapping[777] = 5
#
# df2 = df['a'].apply(lambda x: index_mapping[x]).values
# print(df2)

orders = pd.read_csv("input/order_big.csv")
print(max(orders['user_id']))
print("ok")