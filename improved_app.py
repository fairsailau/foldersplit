import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import io
import base64
from zipfile import ZipFile
import numpy as np
import unittest
import logging
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page to wide mode to use full screen width
st.set_page_config(layout="wide", page_title="Folder Splitting Recommendation Tool")

# Initialize session state for preserving data between reruns
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None
if 'summary_table' not in st.session_state:
    st.session_state.summary_table = None
if 'folder_splits_table' not in st.session_state:
    st.session_state.folder_splits_table = None
if 'users_exceeding' not in st.session_state:
    st.session_state.users_exceeding = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'collaboration_data' not in st.session_state:
    st.session_state.collaboration_data = None
if 'collaboration_analysis' not in st.session_state:
    st.session_state.collaboration_analysis = None

class FolderSplitRecommender:
    """
    A class that analyzes folder ownership data and provides recommendations for splitting content
    based on file count thresholds.
    
    This recommender identifies users who own more than a specified threshold of files (default: 500,000)
    and recommends which folders to split to bring users below this threshold. It assigns excess folders
    to service accounts, ensuring each service account stays under the threshold.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing folder data
        file_threshold (int): Maximum number of files a user should have (default: 500,000)
        users_exceeding (pd.DataFrame): DataFrame of users exceeding the threshold
        recommendations (dict): Dictionary containing recommendations for each user
        collaboration_df (pd.DataFrame, optional): DataFrame containing collaboration data
    """
    
    def __init__(self, df: pd.DataFrame, file_threshold: int = 500000, collaboration_df: Optional[pd.DataFrame] = None):
        """
        Initialize the recommender with the dataframe and threshold.
        
        Args:
            df: DataFrame containing folder data with required columns:
                - Path: Folder path separated by "/"
                - Folder Name: Name of the folder
                - Folder ID: Integer ID of the folder
                - Owner: Email of the user that owns the folder
                - Size (MB): Size of the folder in MB
                - File Count: Number of active files within the folder and all subfolders
                - Level: Folder level in the folder tree hierarchy (1 is root level)
            file_threshold: Maximum number of files a user should have (default: 500,000)
            collaboration_df: Optional DataFrame containing collaboration data with columns:
                - Folder ID: Matches IDs in the main data file
                - Collaboration_ID: Unique identifier for the collaboration
                - Collaborator_login: Email or login of the collaborator
                - Collaborator_ID: Unique identifier for the collaborator
        """
        self.df = df
        self.file_threshold = file_threshold
        # Set service account threshold to 490,000 (hard requirement)
        self.service_account_threshold = 490000
        self.users_exceeding = None
        self.recommendations = {}
        self.collaboration_df = collaboration_df
        self.collaboration_analysis = {}
        
        # Validate input data
        self._validate_input_data()
        
        # Validate collaboration data if provided
        if self.collaboration_df is not None:
            self._validate_collaboration_data()
        
    def _validate_input_data(self) -> None:
        """
        Validate that the input DataFrame has all required columns and proper data types.
        
        Raises:
            ValueError: If required columns are missing or have incorrect data types
        """
        required_columns = ['Path', 'Folder Name', 'Folder ID', 'Owner', 'Size (MB)', 'File Count', 'Level']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check data types
        try:
            # Convert numeric columns to appropriate types if they aren't already
            self.df['File Count'] = pd.to_numeric(self.df['File Count'])
            self.df['Level'] = pd.to_numeric(self.df['Level'])
            self.df['Size (MB)'] = pd.to_numeric(self.df['Size (MB)'])
            
            # Ensure Path and Owner are strings
            self.df['Path'] = self.df['Path'].astype(str)
            self.df['Owner'] = self.df['Owner'].astype(str)
            
        except Exception as e:
            error_msg = f"Error converting data types: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_collaboration_data(self) -> None:
        """
        Validate that the collaboration DataFrame has all required columns and proper data types.
        
        Raises:
            ValueError: If required columns are missing or have incorrect data types
        """
        required_columns = ['Folder ID', 'Collaboration_ID', 'Collaborator_login', 'Collaborator_ID']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in self.collaboration_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns in collaboration data: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check data types
        try:
            # Ensure Folder ID and Collaboration_ID are converted to appropriate types
            self.collaboration_df['Folder ID'] = pd.to_numeric(self.collaboration_df['Folder ID'])
            
            # Ensure Collaborator_login and Collaborator_ID are strings
            self.collaboration_df['Collaborator_login'] = self.collaboration_df['Collaborator_login'].astype(str)
            self.collaboration_df['Collaborator_ID'] = self.collaboration_df['Collaborator_ID'].astype(str)
            
        except Exception as e:
            error_msg = f"Error converting collaboration data types: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # ... [rest of the existing methods] ...
    
    def analyze_collaborations(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze collaboration changes based on folder splitting recommendations.
        
        This method identifies:
        1. For root-level folder splits: No changes to collaborations
        2. For subfolder splits:
           - Collaborators added directly to the folder will see it in their root folders
           - Collaborators added at parent-folder level will lose access
        
        Returns:
            Dictionary containing collaboration analysis for each user:
            {
                user_email: {
                    'collaborators_to_add': [
                        {
                            'folder_id': folder_id,
                            'folder_path': folder_path,
                            'collaborators': [
                                {
                                    'collaborator_id': id,
                                    'collaborator_login': email,
                                    'collaboration_id': collab_id
                                },
                                ...
                            ]
                        },
                        ...
                    ],
                    'collaborators_losing_access': [
                        {
                            'folder_id': folder_id,
                            'folder_path': folder_path,
                            'collaborators': [
                                {
                                    'collaborator_id': id,
                                    'collaborator_login': email,
                                    'collaboration_id': collab_id,
                                    'original_access_level': level
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                },
                ...
            }
        """
        if self.collaboration_df is None:
            logger.warning("No collaboration data provided. Skipping collaboration analysis.")
            return {}
        
        try:
            logger.info("Analyzing collaboration changes...")
            st.write("Analyzing collaboration changes...")
            
            collaboration_analysis = {}
            
            # Create a mapping of folder ID to folder path and level
            folder_info = {}
            for _, row in self.df.iterrows():
                folder_id = row['Folder ID']
                folder_info[folder_id] = {
                    'path': row['Path'],
                    'level': row['Level']
                }
            
            # Create a mapping of folder ID to direct collaborators
            direct_collaborations = {}
            for _, row in self.collaboration_df.iterrows():
                folder_id = row['Folder ID']
                if folder_id not in direct_collaborations:
                    direct_collaborations[folder_id] = []
                
                direct_collaborations[folder_id].append({
                    'collaborator_id': row['Collaborator_ID'],
                    'collaborator_login': row['Collaborator_login'],
                    'collaboration_id': row['Collaboration_ID']
                })
            
            # For each user with recommendations
            for user_email, user_recs in self.recommendations.items():
                # Initialize collaboration analysis for this user
                collaboration_analysis[user_email] = {
                    'collaborators_to_add': [],
                    'collaborators_losing_access': []
                }
                
                # For each recommended split
                for folder in user_recs['recommended_splits']:
                    folder_id = folder['folder_id']
                    folder_path = folder['folder_path']
                    folder_level = folder['level']
                    
                    # Skip if folder ID is not in our folder info mapping
                    if folder_id not in folder_info:
                        continue
                    
                    # Check if this is a root-level folder
                    if folder_level == 1:
                        # Root-level folder, collaborations remain unchanged
                        logger.info(f"Root-level folder: {folder_path} - collaborations remain unchanged")
                        continue
                    
                    # This is a subfolder, analyze collaboration changes
                    collaborators_to_add = []
                    collaborators_losing_access = []
                    
                    # Check for direct collaborators on this folder
                    if folder_id in direct_collaborations:
                        # These collaborators will see the folder in their root folders
                        collaborators_to_add.extend(direct_collaborations[folder_id])
                    
                    # Find parent folders
                    parent_path = '/'.join(folder_path.split('/')[:-1])
                    parent_folders = []
                    
                    for parent_id, info in folder_info.items():
                        if info['path'] == parent_path or folder_path.startswith(f"{info['path']}/"):
                            parent_folders.append(parent_id)
                    
                    # Check for collaborators on parent folders
                    for parent_id in parent_folders:
                        if parent_id in direct_collaborations:
                            # These collaborators will lose access
                            for collaborator in direct_collaborations[parent_id]:
                                # Add original access level information
                                collaborator_with_level = collaborator.copy()
                                collaborator_with_level['original_access_level'] = 'Parent Folder'
                                collaborators_losing_access.append(collaborator_with_level)
                    
                    # Add to analysis if we have collaborators to add or who lose access
                    if collaborators_to_add:
                        collaboration_analysis[user_email]['collaborators_to_add'].append({
                            'folder_id': folder_id,
                            'folder_path': folder_path,
                            'collaborators': collaborators_to_add
                        })
                    
                    if collaborators_losing_access:
                        collaboration_analysis[user_email]['collaborators_losing_access'].append({
                            'folder_id': folder_id,
                            'folder_path': folder_path,
                            'collaborators': collaborators_losing_access
                        })
            
            # Store collaboration analysis
            self.collaboration_analysis = collaboration_analysis
            
            return collaboration_analysis
            
        except Exception as e:
            error_msg = f"Error analyzing collaborations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def get_collaboration_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create tables for collaborators to add and collaborators losing access.
        
        Returns:
            Tuple containing:
            - DataFrame of collaborators to add
            - DataFrame of collaborators losing access
        """
        try:
            # Initialize empty DataFrames
            collaborators_to_add_data = []
            collaborators_losing_access_data = []
            
            # For each user with collaboration analysis
            for user_email, analysis in self.collaboration_analysis.items():
                # Process collaborators to add
                for folder_data in analysis['collaborators_to_add']:
                    folder_id = folder_data['folder_id']
                    folder_path = folder_data['folder_path']
                    
                    for collaborator in folder_data['collaborators']:
                        collaborators_to_add_data.append({
                            'User': user_email,
                            'Folder ID': folder_id,
                            'Folder Path': folder_path,
                            'Collaborator ID': collaborator['collaborator_id'],
                            'Collaborator Login': collaborator['collaborator_login'],
                            'Collaboration ID': collaborator['collaboration_id'],
                            'Action': 'Add to root folders'
                        })
                
                # Process collaborators losing access
                for folder_data in analysis['collaborators_losing_access']:
                    folder_id = folder_data['folder_id']
                    folder_path = folder_data['folder_path']
                    
                    for collaborator in folder_data['collaborators']:
                        collaborators_losing_access_data.append({
                            'User': user_email,
                            'Folder ID': folder_id,
                            'Folder Path': folder_path,
                            'Collaborator ID': collaborator['collaborator_id'],
                            'Collaborator Login': collaborator['collaborator_login'],
                            'Collaboration ID': collaborator['collaboration_id'],
                            'Original Access Level': collaborator['original_access_level']
                        })
            
            # Create DataFrames
            collaborators_to_add_df = pd.DataFrame(collaborators_to_add_data)
            collaborators_losing_access_df = pd.DataFrame(collaborators_losing_access_data)
            
            return collaborators_to_add_df, collaborators_losing_access_df
            
        except Exception as e:
            error_msg = f"Error creating collaboration tables: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise

def main():
    """Main application entry point."""
    st.title("Folder Splitting Recommendation Tool (Optimized Version)")
    
    # Introduction
    st.header("About This Tool")
    st.write("This tool analyzes folder ownership data and provides recommendations for splitting content based on file count thresholds. It identifies users who own more than the specified file threshold and recommends which folders to split to bring users below this threshold.")
    
    st.write("Key Features:")
    st.markdown("""
    1. Correctly calculates total file counts per user
    2. Identifies users exceeding the threshold
    3. Recommends which folders to split to service accounts
    4. Ensures no service account exceeds the threshold
    5. Keeps related folders together when possible
    6. Provides visualizations of the recommendations
    7. Analyzes collaboration changes for folder splits
    """)
    
    # Upload data
    st.header("Upload Data")
    
    # Set threshold with user input
    threshold = st.number_input("File count threshold", min_value=100000, max_value=10000000, value=500000, step=50000)
    st.write(f"Using file count threshold: {threshold:,}")
    st.session_state.threshold = threshold
    
    # Show progress
    progress_bar = st.progress(0)
    st.write("Processing data...")
    
    try:
        # Use file uploader for deployment compatibility
        uploaded_file = st.file_uploader("Upload folder data CSV", type=["csv"])
        
        # Check if file is uploaded
        if uploaded_file is not None:
            # Load data from uploaded file
            df = pd.read_csv(uploaded_file)
            progress_bar.progress(0.1)
            
            # Store threshold in session state
            st.session_state.threshold = threshold
            
            # Process data
            users_exceeding, recommendations, visualizations, summary_table = process_data(df, threshold, progress_bar)
            
            # Store results in session state
            st.session_state.users_exceeding = users_exceeding
            st.session_state.recommendations = recommendations
            st.session_state.visualizations = visualizations
            st.session_state.summary_table = summary_table
            
            # Mark analysis as complete
            st.session_state.analysis_complete = True
            
            # Display results
            display_results()
            
            # Collaboration data upload section
            if st.session_state.analysis_complete:
                st.header("Collaboration Analysis")
                st.write("Upload collaboration data to analyze how folder splits will affect collaborations.")
                
                collaboration_file = st.file_uploader("Upload collaboration data CSV", type=["csv"], key="collaboration_upload")
                
                if collaboration_file is not None:
                    # Load collaboration data
                    collaboration_df = pd.read_csv(collaboration_file)
                    
                    # Store collaboration data in session state
                    st.session_state.collaboration_data = collaboration_df
                    
                    # Process collaboration data
                    process_collaboration_data(df, collaboration_df, threshold)
                    
                    # Display collaboration analysis
                    display_collaboration_analysis()
        else:
            st.info("Please upload a CSV file to begin analysis.")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)

def process_data(df: pd.DataFrame, threshold: int, progress_bar=None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, plt.Figure]], pd.DataFrame]:
    """
    Process the data and generate recommendations.
    
    Args:
        df: DataFrame containing folder data
        threshold: Maximum number of files a user should have
        progress_bar: Optional Streamlit progress bar
    
    Returns:
        Tuple containing:
        - DataFrame of users exceeding the threshold
        - Dictionary of recommendations for each user
        - Dictionary of visualizations for each user
        - Summary table DataFrame
    """
    try:
        # Create the recommender
        recommender = FolderSplitRecommender(df, file_threshold=threshold)
        if progress_bar:
            progress_bar.progress(0.2)
        
        # Calculate user stats
        users_exceeding = recommender.calculate_user_stats()
        if progress_bar:
            progress_bar.progress(0.3)
        
        # If no users exceed the threshold, return early
        if len(users_exceeding) == 0:
            st.success("No users exceed the threshold. No recommendations needed.")
            if progress_bar:
                progress_bar.progress(1.0)
            return users_exceeding, {}, {}, pd.DataFrame()
        
        # Display users exceeding threshold
        st.subheader("Users Exceeding Threshold")
        st.dataframe(users_exceeding)
        
        # Prioritize folders
        recommendations = recommender.prioritize_folders()
        if progress_bar:
            progress_bar.progress(0.6)
        
        # Create visualizations
        visualizations = recommender.visualize_recommendations()
        if progress_bar:
            progress_bar.progress(0.8)
        
        # Store users exceeding in session state
        st.session_state.users_exceeding = users_exceeding
        
        # Store recommendations in session state
        st.session_state.recommendations = recommendations
        
        # Store visualizations in session state
        st.session_state.visualizations = visualizations
        
        # Get summary table
        summary_table = recommender.get_summary_table()
        progress_bar.progress(0.9)
        
        # Store summary table in session state
        st.session_state.summary_table = summary_table
        
        # Get folder splits table
        folder_splits_table = recommender.get_folder_splits_table()
        st.session_state.folder_splits_table = folder_splits_table
        
        # Mark analysis as complete
        st.session_state.analysis_complete = True
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        
        # Return the results
        return users_exceeding, recommendations, visualizations, summary_table
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)  # This will show the full stack trace
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return None, None, None, None

def process_collaboration_data(df: pd.DataFrame, collaboration_df: pd.DataFrame, threshold: int) -> None:
    """
    Process collaboration data and analyze collaboration changes.
    
    Args:
        df: DataFrame containing folder data
        collaboration_df: DataFrame containing collaboration data
        threshold: Maximum number of files a user should have
    """
    try:
        st.write("Processing collaboration data...")
        
        # Create the recommender with collaboration data
        recommender = FolderSplitRecommender(df, file_threshold=threshold, collaboration_df=collaboration_df)
        
        # Calculate user stats if not already done
        if st.session_state.users_exceeding is None:
            recommender.calculate_user_stats()
        
        # Use existing recommendations
        recommender.recommendations = st.session_state.recommendations
        
        # Analyze collaborations
        collaboration_analysis = recommender.analyze_collaborations()
        
        # Store collaboration analysis in session state
        st.session_state.collaboration_analysis = collaboration_analysis
        
        # Get collaboration tables
        collaborators_to_add_df, collaborators_losing_access_df = recommender.get_collaboration_tables()
        
        # Store collaboration tables in session state
        st.session_state.collaborators_to_add_df = collaborators_to_add_df
        st.session_state.collaborators_losing_access_df = collaborators_losing_access_df
        
        st.success("Collaboration analysis complete!")
        
    except Exception as e:
        st.error(f"Error processing collaboration data: {str(e)}")
        st.exception(e)  # This will show the full stack trace
        logger.error(f"Error processing collaboration data: {str(e)}", exc_info=True)

def display_collaboration_analysis() -> None:
    """Display the collaboration analysis results."""
    if 'collaboration_analysis' not in st.session_state or st.session_state.collaboration_analysis is None:
        st.warning("No collaboration analysis to display. Please upload collaboration data.")
        return
    
    st.subheader("Collaboration Analysis Results")
    
    # Display collaborators to add
    if 'collaborators_to_add_df' in st.session_state and not st.session_state.collaborators_to_add_df.empty:
        st.write("### Collaborators to Add")
        st.write("These collaborators need to be added to the following folders after splitting:")
        st.dataframe(st.session_state.collaborators_to_add_df)
        
        # Download CSV button for collaborators to add
        csv_to_add = st.session_state.collaborators_to_add_df.to_csv(index=False)
        b64_to_add = base64.b64encode(csv_to_add.encode()).decode()
        href_to_add = f'<a href="data:file/csv;base64,{b64_to_add}" download="collaborators_to_add.csv">Download Collaborators to Add CSV</a>'
        st.markdown(href_to_add, unsafe_allow_html=True)
    else:
        st.info("No collaborators need to be added to folders after splitting.")
    
    # Display collaborators losing access
    if 'collaborators_losing_access_df' in st.session_state and not st.session_state.collaborators_losing_access_df.empty:
        st.write("### Collaborators Losing Access")
        st.write("These collaborators will lose access to the following folders after splitting:")
        st.dataframe(st.session_state.collaborators_losing_access_df)
        
        # Download CSV button for collaborators losing access
        csv_losing = st.session_state.collaborators_losing_access_df.to_csv(index=False)
        b64_losing = base64.b64encode(csv_losing.encode()).decode()
        href_losing = f'<a href="data:file/csv;base64,{b64_losing}" download="collaborators_losing_access.csv">Download Collaborators Losing Access CSV</a>'
        st.markdown(href_losing, unsafe_allow_html=True)
    else:
        st.info("No collaborators will lose access to folders after splitting.")

def display_results():
    """Display the results of the analysis."""
    # Get the results from session state
    users_exceeding = st.session_state.users_exceeding
    recommendations = st.session_state.recommendations
    visualizations = st.session_state.visualizations
    summary_table = st.session_state.summary_table
    folder_splits_table = st.session_state.folder_splits_table
    
    if not all([users_exceeding is not None, recommendations, visualizations, summary_table is not None]):
        st.warning("No analysis results to display. Please generate recommendations first.")
        return
    
    # Display recommendations for each user
    st.subheader("Recommendations by User")
    
    # Display summary table
    st.dataframe(summary_table)
    
    # Download CSV button for summary table
    csv = summary_table.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="summary.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Display unsplittable folders if any exist
    for user_email, user_recs in recommendations.items():
        if 'unsplittable_folders' in user_recs and user_recs['unsplittable_folders']:
            st.subheader(f"Unsplittable Folders for {user_email}")
            st.write("These folders exceed the threshold but cannot be split because they have no suitable subfolders:")
            
            # Create a DataFrame for unsplittable folders
            unsplittable_df = pd.DataFrame(user_recs['unsplittable_folders'])
            st.dataframe(unsplittable_df[['folder_path', 'file_count', 'reason']])
            
            # Download CSV button for unsplittable folders
            csv_unsplittable = unsplittable_df.to_csv(index=False)
            b64_unsplittable = base64.b64encode(csv_unsplittable.encode()).decode()
            href_unsplittable = f'<a href="data:file/csv;base64,{b64_unsplittable}" download="unsplittable_folders_{user_email.replace("@", "_at_")}.csv">Download Unsplittable Folders CSV</a>'
            st.markdown(href_unsplittable, unsafe_allow_html=True)
    
    # Download CSV button for folder splits table
    st.subheader("Detailed Folder Split Recommendations")
    st.dataframe(folder_splits_table)
    
    csv_splits = folder_splits_table.to_csv(index=False)
    b64_splits = base64.b64encode(csv_splits.encode()).decode()
    href_splits = f'<a href="data:file/csv;base64,{b64_splits}" download="folder_splits.csv">Download Folder Splits CSV</a>'
    st.markdown(href_splits, unsafe_allow_html=True)
    
    # Overall visualizations
    st.subheader("Overall Visualizations")
    
    # Create a bar chart comparing before and after for all users
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter to only original users (not service accounts)
    original_users = summary_table[summary_table['Before Split'] > 0]
    
    users = original_users['User']
    before_values = original_users['Before Split']
    after_values = original_users['After All Splits']
    
    x = np.arange(len(users))
    width = 0.35
    
    ax.bar(x - width/2, before_values, width, label='Before Split')
    ax.bar(x + width/2, after_values, width, label='After All Splits')
    
    # Add threshold line
    threshold = st.session_state.threshold
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:,} files)')
    
    # Add data labels
    for i, v in enumerate(before_values):
        ax.text(i - width/2, v + v*0.02, f'{int(v):,}', ha='center', rotation=90)
    
    for i, v in enumerate(after_values):
        ax.text(i + width/2, v + v*0.02, f'{int(v):,}', ha='center', rotation=90)
    
    ax.set_ylabel('File Count')
    ax.set_title('Before vs. After All Splits by User')
    ax.set_xticks(x)
    ax.set_xticklabels(users)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create a pie chart showing total files moved to service accounts
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get service account data
    service_accounts = summary_table[summary_table['Before Split'] == 0]
    
    if not service_accounts.empty:
        # Group by original user
        service_account_by_user = {}
        for user_email, user_recs in recommendations.items():
            if 'service_accounts' in user_recs:
                service_account_by_user[user_email] = sum(account['total_files'] for account in user_recs['service_accounts'])
        
        # Create pie chart
        labels = list(service_account_by_user.keys())
        sizes = list(service_account_by_user.values())
        
        if sizes:  # Only create pie chart if we have data
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                   colors=plt.cm.tab10.colors[:len(sizes)])
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Total Files Moved to Service Accounts by User')
            st.pyplot(fig)
    
    # Create a bar chart showing number of service accounts needed per user
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get number of service accounts per user
    service_accounts_per_user = original_users['Service Accounts']
    
    ax.bar(users, service_accounts_per_user, color='purple')
    
    # Add data labels
    for i, v in enumerate(service_accounts_per_user):
        if v > 0:
            ax.text(i, v + 0.1, str(int(v)), ha='center')
    
    ax.set_ylabel('Number of Service Accounts')
    ax.set_title('Number of Service Accounts Needed per User')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display detailed recommendations for each user
    for user_email, user_recs in recommendations.items():
        st.write("---")
        st.subheader(f"Detailed Recommendations for {user_email}")
        
        st.write(f"Total file count before split: {user_recs['total_file_count']:,}")
        st.write(f"Total file count after all splits: {user_recs['final_file_count']:,}")
        
        # Calculate files to move
        files_to_move = user_recs['total_recommended_moves']
        st.write(f"Total files to move: {files_to_move:,}")
        
        # Display remaining excess if any
        if user_recs['remaining_excess_files'] > 0:
            st.warning(f"Remaining excess files: {user_recs['remaining_excess_files']:,} (still above threshold)")
            st.write("Note: Some folders were too large to split and were skipped. Consider manual splitting of these folders.")
        
        # Display service account information
        if 'service_accounts' in user_recs and user_recs['service_accounts']:
            st.subheader(f"Service Account Distribution for {user_email}")
            st.write(f"Number of service accounts needed: {len(user_recs['service_accounts'])}")
            
            # Create a table showing service account distribution
            account_data = []
            for account in user_recs['service_accounts']:
                account_data.append({
                    'Service Account': account['account_name'],
                    'Total Files': account['total_files'],
                    'Number of Folders': len(account['folders']),
                    'Percent of Threshold': f"{(account['total_files'] / threshold) * 100:.1f}%"
                })
            
            account_df = pd.DataFrame(account_data)
            st.dataframe(account_df)
            
            # Display folder assignments for each service account
            st.subheader("Folder Assignments to Service Accounts")
            
            # Use a selectbox for service account selection
            # This will maintain state between reruns
            if 'selected_account' not in st.session_state:
                st.session_state.selected_account = {}
            
            # Initialize selected account for this user if not already set
            if user_email not in st.session_state.selected_account:
                st.session_state.selected_account[user_email] = user_recs['service_accounts'][0]['account_name']
            
            # Create columns for the dropdown and display area
            col1, col2 = st.columns([1, 3])
            
            with col1:
                service_account_names = [account['account_name'] for account in user_recs['service_accounts']]
                
                # Use a key that includes the user email to make it unique
                selected_account = st.selectbox(
                    "Select Service Account",
                    service_account_names,
                    key=f"account_select_{user_email}",
                    index=service_account_names.index(st.session_state.selected_account[user_email])
                )
                
                # Update the selected account in session state
                st.session_state.selected_account[user_email] = selected_account
            
            # Display folders for the selected service account
            for account in user_recs['service_accounts']:
                if account['account_name'] == selected_account:
                    with col2:
                        st.write(f"**{account['account_name']}** - {account['total_files']:,} files")
                        
                        # Create a table of folders for this account
                        folder_data = []
                        for folder in account['folders']:
                            folder_data.append({
                                'Folder Name': folder['folder_name'],
                                'Folder Path': folder['folder_path'],
                                'Files to Move': folder['recommended_files_to_move'],
                                'Split Type': 'Partial Split' if folder.get('is_partial_split', False) else 'Complete Split',
                                'Level': folder['level']
                            })
                        
                        folder_df = pd.DataFrame(folder_data)
                        st.dataframe(folder_df, use_container_width=True)
        
        # Display recommended splits
        if len(user_recs['recommended_splits']) > 0:
            st.subheader("Recommended Folder Splits")
            
            # Create a DataFrame from the recommended splits
            splits_df = pd.DataFrame(user_recs['recommended_splits'])
            
            # Create a display DataFrame with renamed columns
            display_df = splits_df.rename(columns={
                'folder_name': 'Folder Name',
                'folder_path': 'Folder Path',
                'level': 'Level',
                'current_file_count': 'Total Files',
                'direct_file_count': 'Direct Files',
                'recommended_files_to_move': 'Files to Move',
                'assigned_to': 'Assigned To'
            })
            
            # Add split type column
            display_df['Split Type'] = splits_df.apply(
                lambda x: 'Partial Split' if x.get('is_partial_split', False) else 'Complete Split',
                axis=1
            )
            
            # Sort by file count (descending)
            display_df = display_df.sort_values('Total Files', ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Display visualizations
            user_viz = visualizations.get(user_email, {})
            
            if user_viz:
                # Create tabs for different visualization categories
                viz_tabs = st.tabs(["File Counts", "Service Accounts", "Additional Visualizations"])
                
                with viz_tabs[0]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(user_viz.get('recommendations', None))
                    with col2:
                        st.pyplot(user_viz.get('current_vs_recommended', None))
                    
                    st.pyplot(user_viz.get('before_after', None))
                
                with viz_tabs[1]:
                    if 'service_account_distribution' in user_viz:
                        st.pyplot(user_viz.get('service_account_distribution', None))

if __name__ == "__main__":
    main()
