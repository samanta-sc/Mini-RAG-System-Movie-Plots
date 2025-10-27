
import pandas as pd
import os
from langchain_core.documents import Document


def dataconverter():
    product_data=pd.read_csv(os.path.join("../data", "flipkart_product_review.csv"), usecols=["product_title","review"])
    df1 = pd.read_csv(os.path.join("../data", "flipkart_product_review_bn_0_to_200.csv"))
    df2 = pd.read_csv(os.path.join("../data", "flipkart_product_review_bn_200_to_450.csv"))


    data=pd.concat([product_data, df1, df2], ignore_index=True)

    product_list = []

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Construct an object with 'product_name' and 'review' attributes
        obj = {
                'product_name': row['product_title'],
                'review': row['review']
            }
        # Append the object to the list
        product_list.append(obj)

        
            
    docs = []
    for entry in product_list:
        metadata = {"product_name": entry['product_name']}
        doc = Document(page_content=entry['review'], metadata=metadata)
        docs.append(doc)
    return docs
