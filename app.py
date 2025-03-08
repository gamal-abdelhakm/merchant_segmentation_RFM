import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import numpy as np

# Load the data from Cleaned_Data_Merchant_Level.csv
data = pd.read_csv('updated.csv')

def get_recommendations_by_id(user_id):
    # Get the cluster of the given User_Id
    user_cluster = data[data['User_Id'] == user_id]['Cluster'].iloc[0]
    
    # Get all users in the same cluster
    users_in_cluster = data[data['Cluster'] == user_cluster]
    
    return get_top_recommendations(users_in_cluster)

def get_recommendations_by_features(features):
    # Extract the feature columns from the dataset
    feature_columns = [col for col in data.columns if col not in ['User_Id', 'Category In English', 'Mer_Id', 'Trx_Rank', 'Cluster']]
    
    # Create a new dataframe with the user's features
    user_features = pd.DataFrame([features], columns=feature_columns)
    
    # Load the KMeans model or recreate it if needed
    # For simplicity, we'll just use the number of clusters from the data
    num_clusters = data['Cluster'].nunique()
    
    # Train KMeans on the feature columns of the original data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data[feature_columns])
    
    # Predict the cluster for the user's features
    user_cluster = kmeans.predict(user_features)[0]
    
    # Get all users in the same cluster
    users_in_cluster = data[data['Cluster'] == user_cluster]
    
    return get_top_recommendations(users_in_cluster)

def get_top_recommendations(users_in_cluster):
    # Group by Category and calculate the total number of transactions for each category
    category_transactions = users_in_cluster.groupby('Category In English')['Trx_Rank'].sum().reset_index()
    
    # Sort categories based on transaction frequency in descending order
    category_transactions = category_transactions.sort_values(by='Trx_Rank', ascending=False)
    
    # Get the top two categories
    top_categories = category_transactions['Category In English'].head(2)
    
    recommendations = []
    
    for category in top_categories:
        # Get all transactions for the top category
        category_data = users_in_cluster[users_in_cluster['Category In English'] == category]
        
        # Group by Merchant and calculate the total number of transactions for each Merchant
        merchant_transactions = category_data.groupby('Mer_Id')['Trx_Rank'].sum().reset_index()
        
        # Join with merchant names if available (assuming there's a column with merchant names)
        if 'Merchant_Name' in data.columns:
            merchant_data = data[['Mer_Id', 'Merchant_Name']].drop_duplicates()
            merchant_transactions = merchant_transactions.merge(merchant_data, on='Mer_Id', how='left')
        
        # Sort merchants based on transaction frequency in descending order
        merchant_transactions = merchant_transactions.sort_values(by='Trx_Rank', ascending=False)
        
        # Get the top five merchants
        if 'Merchant_Name' in data.columns:
            top_merchants = list(zip(merchant_transactions['Mer_Id'].head(5).tolist(), 
                                    merchant_transactions['Merchant_Name'].head(5).tolist()))
        else:
            top_merchants = merchant_transactions['Mer_Id'].head(5).tolist()
        
        recommendations.append((category, top_merchants))
    
    return recommendations

def main():
    st.title("Enhanced Customer Recommendation System")
    
    # Sidebar for mode selection
    st.sidebar.title("Recommendation Mode")
    mode = st.sidebar.radio("Select input method:", ["User ID", "User Features"])
    
    if mode == "User ID":
        st.write("Enter a User_Id to get personalized recommendations.")
        
        # Input user ID
        user_id = st.number_input("Enter User_Id", min_value=0, step=1, value=0)
        
        # Convert user_id to int, as Streamlit returns it as float
        user_id = int(user_id)
        
        if st.button("Get Recommendations by ID"):
            if user_id in data['User_Id'].values:
                recommendations = get_recommendations_by_id(user_id)
                display_recommendations(recommendations, f"User_Id {user_id}")
            else:
                st.error(f"User_Id {user_id} not found in the dataset.")
    
    else:  # User Features mode
        st.write("Enter user features to get personalized recommendations.")
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in ['User_Id', 'Category In English', 'Mer_Id', 'Trx_Rank', 'Cluster']]
        
        # Create input fields for each feature
        feature_values = {}
        
        # Create a two-column layout for features
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(feature_columns):
            # Determine the column to place the feature in
            current_col = col1 if i % 2 == 0 else col2
            
            # Determine if the feature is categorical or numerical
            if data[feature].dtype == 'object' or len(data[feature].unique()) < 10:
                # For categorical features or features with few unique values, use a selectbox
                options = sorted(data[feature].unique().tolist())
                default_value = options[0] if options else None
                feature_values[feature] = current_col.selectbox(f"{feature}:", options, index=0)
            else:
                # For numerical features, use a slider with min and max values from the dataset
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                step = (max_val - min_val) / 100  # 100 steps between min and max
                default_val = float(data[feature].mean())
                feature_values[feature] = current_col.slider(
                    f"{feature}:", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=default_val,
                    step=step
                )
        
        if st.button("Get Recommendations by Features"):
            recommendations = get_recommendations_by_features(feature_values)
            display_recommendations(recommendations, "Custom Features")

def display_recommendations(recommendations, user_identifier):
    st.subheader(f"Recommendations for {user_identifier}:")
    
    for idx, (category, merchants) in enumerate(recommendations, 1):
        st.write(f"**Category {idx}: {category}**")
        
        # Create a container for the merchants
        merchant_container = st.container()
        
        with merchant_container:
            # Check if merchants is a list of tuples (id, name) or just a list of ids
            if merchants and isinstance(merchants[0], tuple):
                for i, (mer_id, name) in enumerate(merchants, 1):
                    merchant_name = name if name else f"Merchant {mer_id}"
                    st.write(f"  {i}. {merchant_name} (ID: {mer_id})")
            else:
                for i, mer_id in enumerate(merchants, 1):
                    st.write(f"  {i}. Merchant ID: {mer_id}")
        
        st.write("")  # Add some spacing between categories

if __name__ == "__main__":
    main()
