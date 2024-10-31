# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Function to determine if a product is successful
def is_successful(group):
    # Drop duplicates based on 'sale_date'
    group = group.drop_duplicates(subset='sale_date').reset_index(drop=True)
    
    # If the group has fewer than 3 records after dropping duplicates, it's not successful
    if len(group) <= 2:
        return pd.DataFrame()  # Not successful

    # Convert sale_date to a numeric year_month for easy month comparison
    group['year_month'] = group['sale_date'].dt.year * 12 + group['sale_date'].dt.month
    
    # Compute the difference between consecutive year_month values
    group['month_diff'] = group['year_month'].diff().fillna(1).astype(int)
    
    # Identify consecutive months (where month_diff == 1)
    consecutive_months = (group['month_diff'] == 1).sum()
    
    # If there are at least 2 consecutive month differences, mark the product as successful
    if consecutive_months >= 2:
        return group[['customer_id', 'product_id', 'sale_gallons', 'sale_date']]
    
    return pd.DataFrame()

# Function to clean and merge dataframes
def clean_data():
    # Load data into dataframes
    sales_df = pd.read_csv('/Users/juanbernal/Documents/Reto/ventas.csv')
    customers_df = pd.read_csv('/Users/juanbernal/Documents/Reto/customers_sampled.csv')
    products_df = pd.read_csv('/Users/juanbernal/Documents/Reto/20230223_productos.csv')

    # Process sales_df #
    sales_df['calmonth'] = pd.to_datetime(sales_df['calmonth'], format='%Y%m').dt.to_period('M')
    
    # Rename columns
    sales_df = sales_df.rename(columns={
        'CustomerId': 'customer_id',
        'material': 'product_id',
        'calmonth': 'sale_date',
        'uni_box': 'sale_gallons'
    })

    # Sort the sales data by customer_id, product_id, and sale_date
    sales_df = sales_df.sort_values(by=['customer_id', 'product_id', 'sale_date'])
    
    # Apply the 'is_successful' function to each group of customer_id and product_id
    successful_sales_df = sales_df.groupby(['customer_id', 'product_id']).apply(is_successful).reset_index(drop=True)

    # Group by customer and product, summing the 'sale_gallons'
    clean_sales_df = successful_sales_df.groupby(['customer_id', 'product_id'])['sale_gallons'].sum().reset_index()

    clean_sales_df['customer_id'] = clean_sales_df['customer_id'].astype(int)
    clean_sales_df['product_id'] = clean_sales_df['product_id'].astype(int)

    # Process customers_df #
    clean_customers_df = customers_df[['CustomerId']].rename(columns={'CustomerId': 'customer_id'})

    # Process products_df #
    products_df = (
        products_df
        .drop(columns=[
            'Material_desc', 'Brand', 'BrandPresRet', 'ProdKey', 'Presentation',
            'Pack', 'Flavor', 'ProductType', 'Container', 'Ncb', 'ProductCategory',
            'SegAg', 'SegDet', 'GlobalSubcategory', 'GlobalFlavor', 'BrandGrouper'
        ])  # Drop unnecessary columns
        .rename(columns={
            'Material': 'product_id',
            'Productos_Por_Empaque': 'product_package_items',
            'Returnability': 'product_returnable',
            'Size': 'product_size',
            'GlobalCategory': 'product_category',
            'MLSize': 'product_size_ml'
        })  # Rename columns
        .dropna()  # Drop rows with missing values
        .replace({
            'product_returnable': {'NO RETORNABLE': 0, 'RETORNABLE': 1},
            'product_size': {'INDIVIDUAL': 0, 'FAMILIAR': 1}
        })  # Replace categorical values
    )
    
    # Get categorical columns and apply one-hot encoding
    categorical_columns = products_df.select_dtypes(include=['object', 'category']).columns
    products_df = pd.get_dummies(products_df, columns=categorical_columns)

    # Replace boolean values with integers
    clean_products_df = products_df.replace({True: 1, False: 0})

    # Merge DataFrames #
    merged_df = clean_sales_df.merge(clean_products_df, on='product_id').merge(clean_customers_df, on='customer_id')

    merged_df.to_csv('merged_df.csv',index=False)

# Function to calculate similarities and probabilities
def evaluate_new_product(data_df, feature_matrix, scaler, new_product):
    # Scale the new product features
    new_product_scaled = scaler.transform(new_product)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(feature_matrix, new_product_scaled)[:, 0]
    
    # Add similarity scores to the dataframe
    data_df['similarity'] = similarities
    
    # Convert similarity to a probability percentage (between 0 and 100)
    data_df['purchase_probability'] = data_df['similarity'] * 100
    
    # Sort the dataframe by similarity, then by sale_gallons in descending order
    sorted_df = data_df[['customer_id', 'product_id', 'sale_gallons', 'similarity', 'purchase_probability']].sort_values(
        by=['similarity', 'sale_gallons', 'customer_id', 'product_id'], 
        ascending=[False, False, True, True]
    )
    
    # Remove repeated customers based on 'customer_id'
    sorted_df = sorted_df.drop_duplicates(subset='customer_id')

    # Save to CSV (overwriting the file)
    sorted_df.to_csv('product_similarity.csv', index=False)
    
    # Display the top 5 similar products
    print("\nTop 5 Similar Products:\n", sorted_df.head())
    
    # Return sorted dataframe and scaled feature matrix
    return sorted_df

# Function to compare the vectors
def compare_vectors(top_n, data_df, new_product, sorted_df):
    # Get top products index
    top_products = sorted_df.head(top_n).index

    # Initialize a list to hold the top products DataFrames
    validate_list = []

    for i in top_products:
        # Append each product row to the list
        validate_list.append(data_df.iloc[[i]])  # Use iloc to get a DataFrame
    
    # Concatenate the list into a single DataFrame
    validate_df = pd.concat(validate_list, ignore_index=True)

     # Create a new product row based on the structure of validate_df
    new_product_row = pd.Series({
        **{col: 0 for col in validate_df.columns[:3]},  # Set customer_id, product_id, and sale_gallons to 0
        **{col: new_product[col].values[0] for col in new_product.columns},  # Copy new product features
        'customer_id': '0000',  # New Customer ID
        'product_id': '0000',    # New Product ID
        'similarity': 0,         # Set similarity to 0
        'purchase_probability': 0  # Set purchase probability to 0
    })

    # Append the new product row to the validate_df
    final_validate_df = pd.concat([pd.DataFrame([new_product_row]), validate_df], ignore_index=True)

    print("\nValidate Top 5 Similar Products (Including New Product)")
    print(final_validate_df.head(top_n+1))

    final_validate_df.to_csv('validation_df.csv',index=False)
    
# Call function to create merged_df.csv
clean_data()

# Load data into dataframes
data_df = pd.read_csv('merged_df.csv')

# Define feature matrix (exclude columns not related to the features)
feature_matrix = data_df.drop(columns=['customer_id', 'product_id', 'sale_gallons']).values

# Normalize features
scaler = StandardScaler()
feature_matrix = scaler.fit_transform(feature_matrix)

# Define new product features
new_product_features = {
    'product_package_items': 6,
    'product_size_ml': 550, 
    'product_returnable': 0,
    'product_size': 0,
    'product_category_AGUA': 1,
    'product_category_BEBIDAS EMERGENTES': 0,
    'product_category_FABS': 0,
    'product_category_LÁCTEOS': 0,
    'product_category_REFRESCOS': 0,
}

# Convert dictionary to DataFrame
new_product = pd.DataFrame([new_product_features])

# Call similarities function
sorted_similarity_df = evaluate_new_product(data_df, feature_matrix, scaler, new_product)

# Compare the new product vector with the top n products
compare_vectors(6, data_df, new_product, sorted_similarity_df)