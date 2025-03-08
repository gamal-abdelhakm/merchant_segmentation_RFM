import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import time

# Set page configuration
st.set_page_config(
    page_title="Advanced Customer Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0277BD;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #0288D1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: white;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #546E7A;
    }
</style>
""", unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('updated.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
data = load_data()

if data is None:
    st.error("Failed to load data. Please check your data file and restart the application.")
    st.stop()

# Extract feature columns for clustering
feature_columns = [col for col in data.columns if col not in ['User_Id', 'Category In English', 'Mer_Id', 'Trx_Rank', 'Cluster']]

# Function to perform clustering if needed
@st.cache_resource
def get_kmeans_model(num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(data[feature_columns])
    return kmeans

# Function to get recommendations based on user ID
def get_recommendations_by_id(user_id, top_n_categories=2, top_n_merchants=5):
    try:
        # Get the cluster of the given User_Id
        user_cluster = data[data['User_Id'] == user_id]['Cluster'].iloc[0]
        
        # Get the user's data for analysis
        user_data = data[data['User_Id'] == user_id]
        
        # Get all users in the same cluster
        users_in_cluster = data[data['Cluster'] == user_cluster]
        
        return get_top_recommendations(users_in_cluster, user_data, top_n_categories, top_n_merchants)
    except IndexError:
        return None, None, None

# Function to get recommendations based on custom features
def get_recommendations_by_features(features, top_n_categories=2, top_n_merchants=5):
    # Create a new dataframe with the user's features
    user_features = pd.DataFrame([features], columns=feature_columns)
    
    # Get the number of clusters from the data
    num_clusters = data['Cluster'].nunique() if 'Cluster' in data.columns else 5
    
    # Get or train KMeans model
    kmeans = get_kmeans_model(num_clusters)
    
    # Predict the cluster for the user's features
    user_cluster = kmeans.predict(user_features)[0]
    
    # Get all users in the same cluster
    users_in_cluster = data[data['Cluster'] == user_cluster]
    
    return get_top_recommendations(users_in_cluster, None, top_n_categories, top_n_merchants)

# Function to generate recommendations
def get_top_recommendations(users_in_cluster, user_data=None, top_n_categories=2, top_n_merchants=5):
    # Group by Category and calculate the total number of transactions for each category
    category_transactions = users_in_cluster.groupby('Category In English')['Trx_Rank'].sum().reset_index()
    
    # Sort categories based on transaction frequency in descending order
    category_transactions = category_transactions.sort_values(by='Trx_Rank', ascending=False)
    
    # Get the top categories
    top_categories = category_transactions['Category In English'].head(top_n_categories).tolist()
    
    recommendations = []
    all_top_merchants = []
    
    for category in top_categories:
        # Get all transactions for the top category
        category_data = users_in_cluster[users_in_cluster['Category In English'] == category]
        
        # Group by Merchant and calculate the total number of transactions for each Merchant
        merchant_transactions = category_data.groupby('Mer_Id')['Trx_Rank'].sum().reset_index()
        
        # Try to get merchant names if available
        if 'Merchant_Name' in data.columns:
            merchant_data = data[['Mer_Id', 'Merchant_Name']].drop_duplicates()
            merchant_transactions = merchant_transactions.merge(merchant_data, on='Mer_Id', how='left')
        
        # Sort merchants based on transaction frequency in descending order
        merchant_transactions = merchant_transactions.sort_values(by='Trx_Rank', ascending=False)
        
        # Get the top merchants
        if 'Merchant_Name' in merchant_transactions.columns:
            top_merchants = list(zip(
                merchant_transactions['Mer_Id'].head(top_n_merchants).tolist(),
                merchant_transactions['Merchant_Name'].head(top_n_merchants).tolist(),
                merchant_transactions['Trx_Rank'].head(top_n_merchants).tolist()
            ))
            # Add to the list of all top merchants
            all_top_merchants.extend([(mid, mname, category, rank) for mid, mname, rank in top_merchants])
        else:
            top_merchants = list(zip(
                merchant_transactions['Mer_Id'].head(top_n_merchants).tolist(),
                merchant_transactions['Trx_Rank'].head(top_n_merchants).tolist()
            ))
            # Add to the list of all top merchants
            all_top_merchants.extend([(mid, f"Merchant {mid}", category, rank) for mid, rank in top_merchants])
        
        recommendations.append((category, top_merchants))
    
    return recommendations, top_categories, all_top_merchants

# Function to visualize user cluster
def visualize_cluster(user_cluster):
    # Get users in the same cluster
    cluster_data = data[data['Cluster'] == user_cluster]
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[feature_columns])
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Cluster': data['Cluster']
    })
    
    # Add a column to highlight the user's cluster
    plot_df['is_user_cluster'] = plot_df['Cluster'] == user_cluster
    
    # Create the plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        opacity=0.6,
        title='Customer Segments (PCA Visualization)',
        color_continuous_scale=px.colors.sequential.Blues,
        height=500
    )
    
    # Add highlighted points for the user's cluster
    fig.add_scatter(
        x=plot_df[plot_df['is_user_cluster']]['PC1'],
        y=plot_df[plot_df['is_user_cluster']]['PC2'],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Your Cluster ({user_cluster})'
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Cluster",
        plot_bgcolor='rgba(240,242,246,0.8)'
    )
    
    return fig

# Function to create category preference visualization
def create_category_chart(recommendations, user_data=None):
    # Extract category data
    categories = []
    values = []
    
    for category, merchants in recommendations:
        # Sum up the transaction ranks for this category
        if isinstance(merchants[0], tuple) and len(merchants[0]) >= 3:
            total_value = sum([m[2] for m in merchants])
        elif isinstance(merchants[0], tuple) and len(merchants[0]) >= 2:
            total_value = sum([m[1] for m in merchants])
        else:
            total_value = len(merchants)
            
        categories.append(category)
        values.append(total_value)
    
    # Create the chart
    fig = px.bar(
        x=categories,
        y=values,
        labels={'x': 'Category', 'y': 'Preference Score'},
        title='Category Preferences',
        text=values,
        color=values,
        color_continuous_scale=px.colors.sequential.Blues,
        height=400
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Category",
        yaxis_title="Preference Score",
        plot_bgcolor='rgba(240,242,246,0.8)'
    )
    
    return fig

# Function to create merchant recommendation visualization
def create_merchant_chart(all_top_merchants):
    # Create DataFrame for plotting
    merchant_df = pd.DataFrame(all_top_merchants, columns=['Merchant_Id', 'Merchant_Name', 'Category', 'Score'])
    
    # Sort by score
    merchant_df = merchant_df.sort_values('Score', ascending=True)
    
    # Create the chart
    fig = px.bar(
        merchant_df,
        y='Merchant_Name',
        x='Score',
        color='Category',
        labels={'Merchant_Name': 'Merchant', 'Score': 'Recommendation Score'},
        title='Top Recommended Merchants',
        orientation='h',
        height=max(400, len(merchant_df) * 30)
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Recommendation Score",
        yaxis_title="",
        plot_bgcolor='rgba(240,242,246,0.8)'
    )
    
    return fig

# Main app
def main():
    # Display header
    st.markdown('<div class="main-header">Advanced Customer Recommendation System</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.image("https://img.icons8.com/color/96/000000/shopping-basket.png", width=80)
    st.sidebar.title("Configuration")
    
    # Mode selection
    mode = st.sidebar.radio("Recommendation Mode:", ["User ID", "Custom Features"])
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        top_n_categories = st.slider("Number of Categories", 1, 5, 2)
        top_n_merchants = st.slider("Merchants per Category", 1, 10, 5)
    
    # Help information
    with st.sidebar.expander("Help & Information"):
        st.markdown("""
        **How to use this app:**
        1. Choose a recommendation mode
        2. Enter your User ID or custom features
        3. Click the "Get Recommendations" button
        4. Explore the personalized recommendations
        
        The system analyzes shopping patterns to suggest relevant merchants based on your profile.
        """)
    
    # Main content area
    if mode == "User ID":
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### Enter User ID")
        st.markdown("Provide a user ID to get personalized recommendations based on their shopping history.")
        
        # Create a row with two columns for input and submit
        col1, col2 = st.columns([3, 1])
        with col1:
            user_id = st.number_input("", min_value=0, step=1, value=data['User_Id'].min() if 'User_Id' in data.columns else 0, help="Enter a valid User ID from the dataset")
        with col2:
            submit_button = st.button("Get Recommendations", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
        if submit_button:
            # Show a spinner while processing
            with st.spinner("Analyzing user profile and generating recommendations..."):
                time.sleep(1)  # Simulating processing time
                
                # Convert user_id to int
                user_id = int(user_id)
                
                # Check if user exists in the dataset
                if user_id in data['User_Id'].values:
                    # Get recommendations
                    recommendations, top_categories, all_top_merchants = get_recommendations_by_id(
                        user_id, top_n_categories, top_n_merchants
                    )
                    
                    # Get user's cluster
                    user_cluster = data[data['User_Id'] == user_id]['Cluster'].iloc[0]
                    
                    # Display results
                    display_recommendations(recommendations, user_id, user_cluster, top_categories, all_top_merchants)
                else:
                    st.error(f"User ID {user_id} not found in the dataset. Please enter a valid User ID.")
    
    else:  # Custom Features mode
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### Custom Feature Selection")
        st.markdown("Adjust the features below to create a custom user profile and get personalized recommendations.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a container for features
        with st.container():
            # Get feature values using a more compact layout with columns
            feature_values = {}
            
            # Determine number of columns based on number of features
            num_cols = 3 if len(feature_columns) > 6 else 2
            cols = st.columns(num_cols)
            
            for i, feature in enumerate(feature_columns):
                col_idx = i % num_cols
                
                # Determine feature type and create appropriate input
                if data[feature].dtype == 'object' or data[feature].nunique() < 10:
                    # For categorical features or features with few unique values
                    options = sorted(data[feature].unique().tolist())
                    feature_values[feature] = cols[col_idx].selectbox(
                        f"{feature}:", 
                        options, 
                        index=0
                    )
                else:
                    # For numerical features
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    step = (max_val - min_val) / 100
                    default_val = float(data[feature].median())
                    feature_values[feature] = cols[col_idx].slider(
                        f"{feature}:", 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default_val,
                        step=step
                    )
        
        # Center the submit button
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            submit_button = st.button("Get Recommendations", type="primary", use_container_width=True)
        
        if submit_button:
            # Show a spinner while processing
            with st.spinner("Analyzing profile and generating recommendations..."):
                time.sleep(1)  # Simulating processing time
                
                # Get recommendations
                recommendations, top_categories, all_top_merchants = get_recommendations_by_features(
                    feature_values, top_n_categories, top_n_merchants
                )
                
                # Predict cluster for the custom features
                kmeans = get_kmeans_model()
                user_features_df = pd.DataFrame([feature_values], columns=feature_columns)
                user_cluster = kmeans.predict(user_features_df)[0]
                
                # Display results
                display_recommendations(recommendations, "Custom Profile", user_cluster, top_categories, all_top_merchants)

def display_recommendations(recommendations, user_identifier, user_cluster, top_categories, all_top_merchants):
    if recommendations is None:
        st.error(f"Could not generate recommendations. Please try a different user ID or feature set.")
        return
    
    # Display success message
    st.success("Recommendations generated successfully!")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Detailed Recommendations", "📈 Analysis"])
    
    with tab1:
        st.markdown(f'<div class="sub-header">Recommendation Dashboard for {user_identifier}</div>', unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">""" + str(user_cluster) + """</div>
                <div class="metric-label">Customer Segment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">""" + str(len(top_categories)) + """</div>
                <div class="metric-label">Recommended Categories</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">""" + str(len(all_top_merchants)) + """</div>
                <div class="metric-label">Recommended Merchants</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display visualizations
        st.markdown('<div class="section-header">Customer Segment Visualization</div>', unsafe_allow_html=True)
        cluster_fig = visualize_cluster(user_cluster)
        st.plotly_chart(cluster_fig, use_container_width=True)
        
        # Split the row for two charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Category Preferences</div>', unsafe_allow_html=True)
            category_fig = create_category_chart(recommendations)
            st.plotly_chart(category_fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Top Merchants</div>', unsafe_allow_html=True)
            merchant_fig = create_merchant_chart(all_top_merchants)
            st.plotly_chart(merchant_fig, use_container_width=True)
    
    with tab2:
        st.markdown(f'<div class="sub-header">Detailed Recommendations for {user_identifier}</div>', unsafe_allow_html=True)
        
        # Display recommendations in a more detailed format
        for idx, (category, merchants) in enumerate(recommendations, 1):
            st.markdown(f'<div class="section-header">Category {idx}: {category}</div>', unsafe_allow_html=True)
            
            # Create a table for merchants
            merchant_data = []
            
            # Check the format of merchants data
            if merchants and isinstance(merchants[0], tuple):
                if len(merchants[0]) >= 3:  # (id, name, rank)
                    for i, (mer_id, name, rank) in enumerate(merchants, 1):
                        merchant_name = name if name else f"Merchant {mer_id}"
                        merchant_data.append({
                            "Rank": i,
                            "Merchant": merchant_name,
                            "ID": mer_id,
                            "Score": rank
                        })
                else:  # (id, name) or (id, rank)
                    for i, merchant_tuple in enumerate(merchants, 1):
                        if len(merchant_tuple) == 2:
                            if isinstance(merchant_tuple[1], (int, float)):  # (id, rank)
                                mer_id, rank = merchant_tuple
                                merchant_name = f"Merchant {mer_id}"
                            else:  # (id, name)
                                mer_id, merchant_name = merchant_tuple
                                rank = i
                            
                            merchant_data.append({
                                "Rank": i,
                                "Merchant": merchant_name,
                                "ID": mer_id,
                                "Score": rank
                            })
            else:
                for i, mer_id in enumerate(merchants, 1):
                    merchant_data.append({
                        "Rank": i,
                        "Merchant": f"Merchant {mer_id}",
                        "ID": mer_id,
                        "Score": i
                    })
            
            # Convert to DataFrame and display as table
            merchant_df = pd.DataFrame(merchant_data)
            st.dataframe(merchant_df, use_container_width=True, hide_index=True)
            
            st.markdown('<hr>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f'<div class="sub-header">Customer Analysis for {user_identifier}</div>', unsafe_allow_html=True)
        
        # Cluster analysis
        st.markdown('<div class="section-header">Segment Profile</div>', unsafe_allow_html=True)
        
        # Get cluster statistics
        cluster_data = data[data['Cluster'] == user_cluster]
        
        # Calculate feature statistics for the cluster
        feature_stats = cluster_data[feature_columns].describe().T[['mean', 'std', 'min', 'max']]
        feature_stats = feature_stats.rename(columns={'mean': 'Average', 'std': 'Std Dev', 'min': 'Minimum', 'max': 'Maximum'})
        
        # Calculate overall statistics
        overall_stats = data[feature_columns].describe().T[['mean', 'std', 'min', 'max']]
        overall_stats = overall_stats.rename(columns={'mean': 'Average', 'std': 'Std Dev', 'min': 'Minimum', 'max': 'Maximum'})
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Segment Avg': feature_stats['Average'],
            'Overall Avg': overall_stats['Average'],
            'Difference': feature_stats['Average'] - overall_stats['Average']
        })
        
        # Format the difference as percentage
        comparison['Difference %'] = (comparison['Difference'] / overall_stats['Average'] * 100).round(2)
        
        # Display the comparison
        st.dataframe(comparison, use_container_width=True)
        
        # Feature importance visualization
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        
        # Sort features by absolute difference
        sorted_features = comparison.sort_values(by='Difference', key=abs, ascending=False).index.tolist()
        
        # Create feature importance chart
        fig = px.bar(
            x=[abs(comparison.loc[f, 'Difference %']) for f in sorted_features],
            y=sorted_features,
            orientation='h',
            labels={'x': 'Absolute Difference %', 'y': 'Feature'},
            title='Feature Importance (Segment vs Overall)',
            color=[comparison.loc[f, 'Difference'] for f in sorted_features],
            color_continuous_scale=px.colors.diverging.RdBu_r,
            height=max(400, len(sorted_features) * 25)
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Absolute Difference %",
            yaxis_title="",
            plot_bgcolor='rgba(240,242,246,0.8)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
