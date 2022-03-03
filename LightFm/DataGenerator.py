import pandas as pd
import numpy as np
import pydbgen
from pydbgen import pydbgen
from faker import Faker
import csv
import datetime
import random


def generate_customers(records, headers):
    fake = Faker('en_US')
    fake2 = Faker('en_GB')
    sex_array = ['M', 'F', 'O']
    with open("input/customer.csv", 'wt') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=headers)
        writer.writeheader()
        for i in range(records):
            sex_index = random.randint(0,2)
            sex = sex_array[sex_index - 1]
            age = random.randint(17,99)
            writer.writerow({
                "Customer_id": i,
                "Name": fake.name(),
                "Sex": sex,
                "Age": age,
                "Job": fake.job(),
                "Phone Number": fake2.phone_number(),
                #"Address": fake.address(),
                "Zip Code": fake.zipcode(),
                "City": fake.city(),
                "State": fake.state(),
                "Country": fake.country()
            })

def generate_orders():
    orders1 = pd.read_csv("input/order_big.csv")
    orders2 = pd.read_csv("input/order_products__train.csv")
    order = orders2.merge(orders1)
    order_df = order[["user_id", "product_id"]]
    order_df["order_count"] = 1
    max_user_id = max(order_df["user_id"])
    print(max_user_id)
    orders = order_df.groupby(["user_id", "product_id"], as_index=False)["order_count"].sum()
    orders.to_csv("input/order.csv", sep=',', encoding='utf-8', index=False)

    print(order.shape)

if __name__ == '__main__':
    records = 206209
    headers = ["Customer_id", "Name", "Sex", "Age", "Job", "Phone Number",
               "Zip Code", "City", "State", "Country"]
    generate_customers(records, headers)
    generate_orders()
    print("CSV generation complete!")
