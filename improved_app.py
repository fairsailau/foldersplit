import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import io
import base64
from zipfile import ZipFile
import numpy as np
import unittest
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Set
import altair as alt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page to wide mode to use full screen width
st.set_page_config(layout="wide", page_title="Folder Splitting Recommendation Tool", page_icon="📁")

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
if 'filtered_folders' not in st.session_state:
    st.session_state.filtered_folders = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

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
    """
    
    def __init__(self, df: pd.DataFrame, file_threshold: int = 500000):
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
        """
        self.df = df
        self.file_threshold = file_threshold
        # Set service account threshold to 490,000 (hard requirement)
        self.service_account_threshold = 490000
        self.users_exceeding = None
        self.recommendations = {}
        
        # Validate input data
        self._validate_input_data()
        
    def _validate_input_data(self) -> None:
        """
        Validate that the input DataFrame has all required columns and proper data types.
        """
        required_columns = ['Path', 'Folder Name', 'Folder ID', 'Owner', 'Size (MB)', 'File Count', 'Level']
        
        # Check if all required columns exist
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert columns to appropriate data types
        try:
            self.df['Folder ID'] = self.df['Folder ID'].astype(int)
            self.df['File Count'] = self.df['File Count'].astype(int)
            self.df['Size (MB)'] = self.df['Size (MB)'].astype(float)
            self.df['Level'] = self.df['Level'].astype(int)
        except Exception as e:
            raise ValueError(f"Error converting data types: {str(e)}")
    
    def identify_users_exceeding_threshold(self) -> pd.DataFrame:
        """
        Identify users who exceed the file threshold.
        
        Returns:
            DataFrame containing users exceeding the threshold with their total file counts
        """
        logger.info("Identifying users exceeding threshold...")
        
        try:
            # Group by Owner and sum File Count
            user_totals = self.df.groupby('Owner')['File Count'].sum().reset_index()
            user_totals.columns = ['Owner', 'total_file_count']
            
            # Filter to users exceeding threshold
            users_exceeding = user_totals[user_totals['total_file_count'] > self.file_threshold]
            
            if users_exceeding.empty:
                logger.info("No users found exceeding the threshold.")
                return pd.DataFrame()
            
            # Sort by total file count (descending)
            users_exceeding = users_exceeding.sort_values('total_file_count', ascending=False)
            
            logger.info(f"Found {len(users_exceeding)} users exceeding the threshold.")
            self.users_exceeding = users_exceeding
            return users_exceeding
            
        except Exception as e:
            error_msg = f"Error identifying users exceeding threshold: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def identify_nested_folders(self) -> None:
        """
        Identify nested folder relationships and calculate direct file counts.
        
        This method adds a 'direct_file_count' column to the DataFrame, which represents
        the number of files directly in the folder (excluding subfolders).
        """
        logger.info("Identifying nested folder relationships...")
        
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df_copy = self.df.copy()
            
            # Sort by path length to ensure parent folders are processed before children
            df_copy['path_length'] = df_copy['Path'].str.len()
            df_copy = df_copy.sort_values('path_length')
            
            # Initialize direct file count as total file count
            df_copy['direct_file_count'] = df_copy['File Count']
            
            # For each folder, subtract its file count from all parent folders
            for i, row in df_copy.iterrows():
                folder_path = row['Path']
                file_count = row['File Count']
                
                # Find all parent folders
                for j, parent_row in df_copy.iterrows():
                    parent_path = parent_row['Path']
                    
                    # Check if this is a parent folder (folder_path starts with parent_path)
                    if i != j and folder_path.startswith(parent_path) and folder_path != parent_path:
                        # Subtract this folder's file count from the parent's direct file count
                        df_copy.at[j, 'direct_file_count'] -= file_count
            
            # Update the original DataFrame with the calculated direct file counts
            self.df['direct_file_count'] = df_copy['direct_file_count']
            
            logger.info("Nested folder relationships identified successfully.")
            
        except Exception as e:
            error_msg = f"Error identifying nested folders: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def _is_subfolder_of_any(self, folder_path: str, selected_folders: List[str]) -> bool:
        """
        Check if a folder is a subfolder of any folder in the selected_folders list.
        
        Args:
            folder_path: Path of the folder to check
            selected_folders: List of folder paths that have already been selected for splitting
            
        Returns:
            True if the folder is a subfolder of any selected folder, False otherwise
        """
        # Normalize paths to ensure consistent comparison
        folder_path = folder_path.strip('/')
        
        for selected_folder in selected_folders:
            # Normalize selected folder path
            selected_folder = selected_folder.strip('/')
            
            # Check if folder_path starts with selected_folder followed by a slash
            # This ensures it's a proper subfolder relationship at any level
            if (folder_path == selected_folder or 
                folder_path.startswith(selected_folder + '/') or 
                '/' + selected_folder + '/' in '/' + folder_path):
                return True
        return False
    
    def _is_parent_of_any(self, folder_path: str, selected_folders: List[str]) -> bool:
        """
        Check if a folder is a parent of any folder in the selected_folders list.
        
        Args:
            folder_path: Path of the folder to check
            selected_folders: List of folder paths that have already been selected for splitting
            
        Returns:
            True if the folder is a parent of any selected folder, False otherwise
        """
        # Normalize path to ensure consistent comparison
        folder_path = folder_path.strip('/')
        
        for selected_folder in selected_folders:
            # Normalize selected folder path
            selected_folder = selected_folder.strip('/')
            
            # Check if selected_folder starts with folder_path followed by a slash
            # This ensures it's a proper parent relationship at any level
            if (selected_folder == folder_path or 
                selected_folder.startswith(folder_path + '/') or 
                '/' + folder_path + '/' in '/' + selected_folder):
                return True
        return False
    
    def assign_to_service_accounts(self, user_email: str, folders_to_split: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign folders to service accounts, keeping related folders together when possible
        and ensuring each service account stays under the threshold.
        
        This method groups folders by common parent paths to keep related folders together,
        then assigns them to service accounts while ensuring no account exceeds the threshold.
        
        Args:
            user_email: Email of the user whose folders need to be split
            folders_to_split: List of folders to split
            
        Returns:
            List of service accounts with assigned folders
        """
        logger.info(f"Assigning folders to service accounts for user: {user_email}")
        
        try:
            # Group folders by common parent paths
            folder_groups = {}
            
            for folder in folders_to_split:
                folder_path = folder['folder_path']
                
                # Extract parent path components
                path_parts = folder_path.split('/')
                
                # Use the first two levels as the group key
                group_key = '/'.join(path_parts[:min(2, len(path_parts))])
                
                if group_key not in folder_groups:
                    folder_groups[group_key] = []
                
                folder_groups[group_key].append(folder)
            
            # Initialize service accounts
            service_accounts = []
            current_account = {
                'account_name': 'service_account_1',
                'folders': [],
                'total_files': 0
            }
            
            # Track which folder is assigned to which account
            folder_to_account = {}
            
            # Assign folders to service accounts, keeping related folders together
            for group_key, group_folders in folder_groups.items():
                # Sort folders within group by path length (ascending) to process parents before children
                group_folders.sort(key=lambda x: len(x['folder_path']))
                
                for folder in group_folders:
                    folder_path = folder['folder_path']
                    
                    # Check if this folder is a parent or child of any already assigned folder
                    assigned_to_account = None
                    
                    # Check if any parent folder has already been assigned
                    for path in folder_to_account:
                        # Normalize paths for comparison
                        norm_folder_path = folder_path.strip('/')
                        norm_path = path.strip('/')
                        
                        # Check if this folder is a child of an already assigned folder
                        if (norm_folder_path == norm_path or 
                            norm_folder_path.startswith(norm_path + '/') or 
                            '/' + norm_path + '/' in '/' + norm_folder_path):
                            # This folder is a child of an already assigned folder
                            assigned_to_account = folder_to_account[path]
                            logger.info(f"Folder {folder_path} is a child of {path}, assigning to same account: {assigned_to_account}")
                            break
                    
                    # Check if any child folder has already been assigned
                    if assigned_to_account is None:
                        for path in folder_to_account:
                            # Normalize paths for comparison
                            norm_folder_path = folder_path.strip('/')
                            norm_path = path.strip('/')
                            
                            # Check if this folder is a parent of an already assigned folder
                            if (norm_path == norm_folder_path or 
                                norm_path.startswith(norm_folder_path + '/') or 
                                '/' + norm_folder_path + '/' in '/' + norm_path):
                                # This folder is a parent of an already assigned folder
                                assigned_to_account = folder_to_account[path]
                                logger.info(f"Folder {folder_path} is a parent of {path}, assigning to same account: {assigned_to_account}")
                                break
                    
                    # If this folder needs to be assigned to a specific account due to parent/child relationship
                    if assigned_to_account is not None:
                        # Find the account
                        target_account = None
                        for account in service_accounts:
                            if account['account_name'] == assigned_to_account:
                                target_account = account
                                break
                        
                        if target_account:
                            # Add folder to the target account
                            folder['assigned_to'] = target_account['account_name']
                            target_account['folders'].append(folder)
                            target_account['total_files'] += folder['recommended_files_to_move']
                            folder_to_account[folder_path] = target_account['account_name']
                            continue
                    
                    # If no parent/child relationship constraints, proceed with normal assignment
                    # Check if adding this folder would exceed the threshold for current account
                    if current_account['total_files'] + folder['recommended_files_to_move'] > self.service_account_threshold:
                        # Only add the current account to service_accounts if it has folders
                        if current_account['folders']:
                            service_accounts.append(current_account)
                        
                        # Create a new account
                        current_account = {
                            'account_name': f'service_account_{len(service_accounts) + 1}',
                            'folders': [],
                            'total_files': 0
                        }
                    
                    # Add folder to current account
                    folder['assigned_to'] = current_account['account_name']
                    current_account['folders'].append(folder)
                    current_account['total_files'] += folder['recommended_files_to_move']
                    folder_to_account[folder_path] = current_account['account_name']
            
            # Add the last account if it has folders
            if current_account['folders']:
                service_accounts.append(current_account)
            
            # Renumber service accounts to ensure no gaps in numbering
            for i, account in enumerate(service_accounts):
                account['account_name'] = f'service_account_{i+1}'
                # Update assigned_to in folders
                for folder in account['folders']:
                    folder['assigned_to'] = account['account_name']
            
            logger.info(f"Created {len(service_accounts)} service accounts for user {user_email}")
            return service_accounts
            
        except Exception as e:
            error_msg = f"Error assigning folders to service accounts: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def prioritize_folders(self) -> Dict[str, Any]:
        """
        Prioritize folders for splitting based on file count and assign to service accounts.
        
        This method uses a hybrid approach:
        1. First adds large folders until we get close to the threshold
        2. Then switches to smaller folders to fine-tune and get as close as possible to the threshold
        3. Stops when adding another folder would bring the user below the threshold
        
        Returns:
            Dictionary of recommendations for each user
        """
        logger.info("Prioritizing folders for splitting...")
        st.write("Prioritizing folders for splitting...")
        
        try:
            # First, identify nested folder relationships and calculate direct file counts
            self.identify_nested_folders()
            
            self.recommendations = {}
            
            # For each user exceeding the threshold
            for _, user_row in self.users_exceeding.iterrows():
                user_email = user_row['Owner']
                total_file_count = user_row['total_file_count']
                excess_files = total_file_count - self.file_threshold
                
                logger.info(f"Processing recommendations for user: {user_email}")
                st.write(f"Processing recommendations for user: {user_email}")
                st.write(f"Total file count (from level 1 folders): {total_file_count:,}, Excess files: {excess_files:,}")
                
                # Get all folders owned by this user
                user_folders = self.df[self.df['Owner'] == user_email].copy()
                
                # Initialize recommendations for this user
                recommendations = {
                    'user_email': user_email,
                    'total_file_count': int(total_file_count),
                    'excess_files': int(excess_files),
                    'recommended_splits': [],
                    'unsplittable_folders': []  # New list to track folders that can't be split
                }
                
                # Track remaining files to be split
                remaining_files = total_file_count
                
                # Track which folders have been selected for splitting
                selected_folder_paths = []
                
                # Find suitable candidates across all levels
                candidates = []
                
                # Process all folders regardless of level
                for _, folder in user_folders.iterrows():
                    folder_path = folder['Path']
                    folder_file_count = folder['File Count']
                    
                    # Skip folders that are too small to be worth splitting
                    if folder_file_count < 10000:
                        continue
                    
                    # Skip if this folder is a subfolder of any already selected folder
                    if self._is_subfolder_of_any(folder_path, selected_folder_paths):
                        continue
                    
                    # Skip if this folder is a parent of any already selected folder
                    if self._is_parent_of_any(folder_path, selected_folder_paths):
                        continue
                    
                    # Track folders that exceed the threshold instead of doing partial splits
                    if folder_file_count > self.file_threshold:
                        logger.info(f"Unsplittable folder: {folder_path} ({folder_file_count:,} files) - exceeds threshold")
                        st.write(f"Unsplittable folder: {folder_path} ({folder_file_count:,} files) - exceeds threshold")
                        
                        # Add to unsplittable folders list
                        recommendations['unsplittable_folders'].append({
                            'folder_path': folder_path,
                            'folder_name': folder['Folder Name'],
                            'folder_id': folder['Folder ID'],
                            'level': folder['Level'],
                            'file_count': int(folder_file_count),
                            'reason': "Exceeds threshold and cannot be split (no suitable subfolders)"
                        })
                        continue
                    
                    # Check if this folder is a good candidate (≤ threshold files)
                    # Calculate how much this split would reduce the total
                    new_total = remaining_files - folder_file_count
                    
                    # Add to candidates
                    candidates.append({
                        'folder_path': folder_path,
                        'folder_name': folder['Folder Name'],
                        'folder_id': folder['Folder ID'],
                        'level': folder['Level'],
                        'current_file_count': int(folder_file_count),
                        'direct_file_count': int(folder['direct_file_count']),
                        'new_total': new_total
                    })
                
                # MODIFIED APPROACH:
                # Only use Phase 1 - Add large folders first until we get close to the threshold
                candidates_large_first = sorted(candidates, key=lambda x: x['current_file_count'], reverse=True)
                
                # Track folders added in phase 1 (large folders first)
                selected_folders = []
                remaining_files_after_selection = remaining_files
                
                for candidate in candidates_large_first:
                    # Skip if this folder is a subfolder of any already selected folder
                    if self._is_subfolder_of_any(candidate['folder_path'], selected_folder_paths):
                        logger.info(f"Skipping folder: {candidate['folder_path']} (Level {candidate['level']}) - is a subfolder of an already selected folder")
                        continue
                    
                    # Skip if this folder is a parent of any already selected folder
                    if self._is_parent_of_any(candidate['folder_path'], selected_folder_paths):
                        logger.info(f"Skipping folder: {candidate['folder_path']} (Level {candidate['level']}) - is a parent of an already selected folder")
                        continue
                    
                    # Add this candidate to selected folders
                    selected_folders.append(candidate)
                    
                    # Update remaining files
                    remaining_files_after_selection = candidate['new_total']
                    logger.info(f"Added folder: {candidate['folder_path']} (Level {candidate['level']}), file count: {candidate['current_file_count']:,}, new total: {remaining_files_after_selection:,}")
                    st.write(f"  Added folder: {candidate['folder_path']} (Level {candidate['level']}), file count: {candidate['current_file_count']:,}, new total: {remaining_files_after_selection:,}")
                    
                    # Add this folder to the selected folders list
                    selected_folder_paths.append(candidate['folder_path'])
                
                # Add selected folders to recommendations
                for folder in selected_folders:
                    recommendations['recommended_splits'].append({
                        'folder_path': folder['folder_path'],
                        'folder_name': folder['folder_name'],
                        'folder_id': folder['folder_id'],
                        'level': folder['level'],
                        'recommended_files_to_move': folder['current_file_count'],
                        'new_total_after_split': folder['new_total']
                    })
                
                # Final remaining files after selection
                remaining_files = remaining_files_after_selection
                st.write(f"After processing all candidates, remaining files: {remaining_files:,}")
                
                # Assign folders to service accounts
                service_accounts = self.assign_to_service_accounts(user_email, recommendations['recommended_splits'])
                recommendations['service_accounts'] = service_accounts
                
                # Calculate total files moved to service accounts
                total_files_moved = sum(account['total_files'] for account in service_accounts)
                
                # Calculate the correct final file count for the user
                # This should be the original total minus the files moved to service accounts
                final_file_count = total_file_count - total_files_moved
                
                # Add summary to recommendations
                recommendations['summary'] = {
                    'original_file_count': int(total_file_count),
                    'final_file_count': int(final_file_count),
                    'total_files_moved': int(total_files_moved),
                    'service_account_count': len(service_accounts)
                }
                
                # Store recommendations for this user
                self.recommendations[user_email] = recommendations
            
            logger.info("Folder prioritization completed successfully.")
            return self.recommendations
            
        except Exception as e:
            error_msg = f"Error prioritizing folders: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def generate_visualizations(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate visualizations for the recommendations.
        
        Returns:
            Dictionary of visualizations for each user
        """
        logger.info("Generating visualizations...")
        
        try:
            visualizations = {}
            
            for user_email, recommendations in self.recommendations.items():
                user_visualizations = {}
                
                # Get data for this user
                original_file_count = recommendations['total_file_count']
                final_file_count = recommendations['summary']['final_file_count']
                service_accounts = recommendations['service_accounts']
                
                # Create before/after comparison chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Bar chart data
                labels = ['Before Split', 'After Split']
                values = [original_file_count, final_file_count]
                
                # Create bars
                ax.bar(labels, values, color=['#1f77b4', '#2ca02c'])
                
                # Add threshold line
                ax.axhline(y=self.file_threshold, color='red', linestyle='--', label=f'Threshold ({self.file_threshold:,} files)')
                
                # Add data labels
                for i, v in enumerate(values):
                    ax.text(i, v + v*0.02, f'{int(v):,}', ha='center')
                
                # Add labels and title
                ax.set_ylabel('File Count')
                ax.set_title(f'Before vs. After Split for {user_email}')
                ax.legend()
                
                # Save figure
                user_visualizations['before_after_comparison'] = fig
                
                # Create service account distribution chart
                if service_accounts:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Bar chart data
                    account_names = [account['account_name'] for account in service_accounts]
                    account_file_counts = [account['total_files'] for account in service_accounts]
                    
                    # Create bars
                    ax.bar(account_names, account_file_counts, color='#ff7f0e')
                    
                    # Add threshold line
                    ax.axhline(y=self.file_threshold, color='red', linestyle='--', label=f'Threshold ({self.file_threshold:,} files)')
                    
                    # Add data labels
                    for i, v in enumerate(account_file_counts):
                        ax.text(i, v + v*0.02, f'{int(v):,}', ha='center')
                    
                    # Add labels and title
                    ax.set_ylabel('File Count')
                    ax.set_title(f'Service Account Distribution for {user_email}')
                    ax.set_xticklabels(account_names, rotation=45)
                    ax.legend()
                    
                    # Save figure
                    user_visualizations['service_account_distribution'] = fig
                
                # Store visualizations for this user
                visualizations[user_email] = user_visualizations
            
            logger.info("Visualizations generated successfully.")
            return visualizations
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table of the recommendations.
        
        Returns:
            DataFrame containing summary information for all users and service accounts
        """
        logger.info("Generating summary table...")
        
        try:
            # Create a list to store summary data
            summary_data = []
            
            # Add data for each user
            for user_email, recommendations in self.recommendations.items():
                # Get summary data
                original_file_count = recommendations['total_file_count']
                final_file_count = recommendations['summary']['final_file_count']
                total_files_moved = recommendations['summary']['total_files_moved']
                
                # Add user row
                summary_data.append({
                    'User': user_email,
                    'Account Type': 'Original User',
                    'Before Split': original_file_count,
                    'After All Splits': final_file_count,
                    'Files to Move': total_files_moved,
                    'Service Accounts Created': recommendations['summary']['service_account_count']
                })
                
                # Add service account rows
                for account in recommendations['service_accounts']:
                    summary_data.append({
                        'User': account['account_name'],
                        'Account Type': 'Service Account',
                        'Before Split': 0,
                        'After All Splits': account['total_files'],
                        'Files to Move': account['total_files'],
                        'Service Accounts Created': 0
                    })
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            logger.info("Summary table generated successfully.")
            return summary_df
            
        except Exception as e:
            error_msg = f"Error generating summary table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def generate_folder_splits_table(self) -> pd.DataFrame:
        """
        Generate a detailed table of folder splits.
        
        Returns:
            DataFrame containing detailed information about all folder splits
        """
        logger.info("Generating folder splits table...")
        
        try:
            # Create a list to store folder split data
            folder_splits_data = []
            
            # Add data for each user
            for user_email, recommendations in self.recommendations.items():
                # Add rows for each recommended split
                for folder in recommendations['recommended_splits']:
                    folder_splits_data.append({
                        'User': user_email,
                        'Folder Path': folder['folder_path'],
                        'Folder Name': folder['folder_name'],
                        'Folder ID': folder['folder_id'],
                        'Level': folder['level'],
                        'Files to Move': folder['recommended_files_to_move'],
                        'New Total After Split': folder['new_total_after_split'],
                        'Assigned To': folder['assigned_to'] if 'assigned_to' in folder else 'Not Assigned'
                    })
            
            # Create DataFrame
            folder_splits_df = pd.DataFrame(folder_splits_data)
            
            logger.info("Folder splits table generated successfully.")
            return folder_splits_df
            
        except Exception as e:
            error_msg = f"Error generating folder splits table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def analyze(self) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame]:
        """
        Analyze the data and generate recommendations.
        
        Returns:
            Tuple containing:
            - DataFrame of users exceeding the threshold
            - Dictionary of recommendations for each user
            - Dictionary of visualizations for each user
            - Summary table as a DataFrame
        """
        logger.info("Starting analysis...")
        
        try:
            # Identify users exceeding the threshold
            users_exceeding = self.identify_users_exceeding_threshold()
            
            if users_exceeding.empty:
                logger.info("No users found exceeding the threshold. Analysis complete.")
                return pd.DataFrame(), {}, {}, pd.DataFrame()
            
            # Prioritize folders for splitting
            recommendations = self.prioritize_folders()
            
            # Generate visualizations
            visualizations = self.generate_visualizations()
            
            # Generate summary table
            summary_table = self.generate_summary_table()
            
            # Generate folder splits table
            folder_splits_table = self.generate_folder_splits_table()
            
            logger.info("Analysis completed successfully.")
            return users_exceeding, recommendations, visualizations, summary_table, folder_splits_table
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.exception(e)  # This will show the full stack trace
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            return None, None, None, None, None

def display_results():
    """Display the results of the analysis."""
    # Get the results from session state
    users_exceeding = st.session_state.users_exceeding
    recommendations = st.session_state.recommendations
    visualizations = st.session_state.visualizations
    summary_table = st.session_state.summary_table
    folder_splits_table = st.session_state.folder_splits_table
    processing_time = st.session_state.processing_time
    
    if not all([users_exceeding is not None, recommendations, visualizations, summary_table is not None]):
        st.warning("No analysis results to display. Please generate recommendations first.")
        return
    
    # Display processing time
    st.info(f"Analysis completed in {processing_time:.2f} seconds")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Detailed Recommendations", "Unsplittable Folders", "Visualizations"])
    
    with tab1:
        # Display recommendations for each user
        st.subheader("Recommendations by User")
        
        # Display summary table with interactive features
        st.dataframe(summary_table, use_container_width=True)
        
        # Download CSV button for summary table
        csv = summary_table.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button(
            label="Download Summary CSV",
            data=csv,
            file_name="summary.csv",
            mime="text/csv"
        )
        
        # Overall visualizations using Plotly
        st.subheader("Overall Visualizations")
        
        # Create a bar chart comparing before and after for all users
        original_users = summary_table[summary_table['Account Type'] == 'Original User']
        
        if not original_users.empty:
            fig = px.bar(
                original_users,
                x='User',
                y=['Before Split', 'After All Splits'],
                barmode='group',
                title='Before vs. After All Splits by User',
                labels={'value': 'File Count', 'variable': 'Status'},
                height=500
            )
            
            # Add threshold line
            threshold = st.session_state.threshold
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(original_users)-0.5,
                y0=threshold,
                y1=threshold,
                line=dict(color="red", width=2, dash="dash"),
                name=f"Threshold ({threshold:,} files)"
            )
            
            # Add annotation for threshold
            fig.add_annotation(
                x=len(original_users)-1,
                y=threshold,
                text=f"Threshold: {threshold:,} files",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # NEW: Download CSV button for folder splits table
        st.subheader("Detailed Folder Split Recommendations")
        
        # Add filtering options
        st.write("Filter recommendations:")
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by user
            users = folder_splits_table['User'].unique()
            selected_user = st.selectbox("Select User", options=['All Users'] + list(users))
        
        with col2:
            # Filter by level
            levels = folder_splits_table['Level'].unique()
            selected_level = st.selectbox("Select Folder Level", options=['All Levels'] + sorted(list(levels)))
        
        # Apply filters
        filtered_df = folder_splits_table.copy()
        if selected_user != 'All Users':
            filtered_df = filtered_df[filtered_df['User'] == selected_user]
        if selected_level != 'All Levels':
            filtered_df = filtered_df[filtered_df['Level'] == selected_level]
        
        # Store filtered data in session state
        st.session_state.filtered_folders = filtered_df
        
        # Display filtered data
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download filtered data
        csv_splits = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Recommendations CSV",
            data=csv_splits,
            file_name="folder_splits_filtered.csv",
            mime="text/csv"
        )
        
        # Display service account distribution
        st.subheader("Service Account Distribution")
        
        # Get service account data
        service_accounts = summary_table[summary_table['Account Type'] == 'Service Account']
        
        if not service_accounts.empty:
            # Create interactive bar chart
            fig = px.bar(
                service_accounts,
                x='User',
                y='After All Splits',
                title='Files Assigned to Service Accounts',
                labels={'After All Splits': 'File Count', 'User': 'Service Account'},
                height=500,
                color='After All Splits',
                color_continuous_scale='Viridis'
            )
            
            # Add threshold line
            threshold = st.session_state.service_account_threshold
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(service_accounts)-0.5,
                y0=threshold,
                y1=threshold,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add annotation for threshold
            fig.add_annotation(
                x=len(service_accounts)-1,
                y=threshold,
                text=f"Service Account Threshold: {threshold:,} files",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Display unsplittable folders if any exist
        st.subheader("Unsplittable Folders")
        
        has_unsplittable = False
        for user_email, user_recs in recommendations.items():
            if 'unsplittable_folders' in user_recs and user_recs['unsplittable_folders']:
                has_unsplittable = True
                st.write(f"### Unsplittable Folders for {user_email}")
                st.write("These folders exceed the threshold but cannot be split because they have no suitable subfolders:")
                
                # Create a DataFrame for unsplittable folders
                unsplittable_df = pd.DataFrame(user_recs['unsplittable_folders'])
                
                # Create interactive table
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=["Folder Path", "File Count", "Reason"],
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[
                            unsplittable_df['folder_path'],
                            unsplittable_df['file_count'].apply(lambda x: f"{x:,}"),
                            unsplittable_df['reason']
                        ],
                        fill_color='lavender',
                        align='left'
                    )
                )])
                
                fig.update_layout(
                    title=f"Unsplittable Folders for {user_email}",
                    height=400 + (len(unsplittable_df) * 25)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download CSV button for unsplittable folders
                csv_unsplittable = unsplittable_df.to_csv(index=False)
                st.download_button(
                    label=f"Download Unsplittable Folders for {user_email}",
                    data=csv_unsplittable,
                    file_name=f"unsplittable_folders_{user_email.replace('@', '_at_')}.csv",
                    mime="text/csv"
                )
        
        if not has_unsplittable:
            st.info("No unsplittable folders found in the analysis.")
    
    with tab4:
        # Create interactive visualizations
        st.subheader("Interactive Visualizations")
        
        # Select user for visualization
        users = list(recommendations.keys())
        if users:
            selected_viz_user = st.selectbox("Select User for Visualization", options=users)
            
            if selected_viz_user in recommendations:
                user_recs = recommendations[selected_viz_user]
                
                # Before/After comparison using Plotly
                before_after_data = pd.DataFrame({
                    'Stage': ['Before Split', 'After Split'],
                    'File Count': [
                        user_recs['total_file_count'],
                        user_recs['summary']['final_file_count']
                    ]
                })
                
                fig = px.bar(
                    before_after_data,
                    x='Stage',
                    y='File Count',
                    title=f'Before vs. After Split for {selected_viz_user}',
                    color='Stage',
                    text_auto=True,
                    height=500
                )
                
                # Add threshold line
                threshold = st.session_state.threshold
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=1.5,
                    y0=threshold,
                    y1=threshold,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Add annotation for threshold
                fig.add_annotation(
                    x=1,
                    y=threshold,
                    text=f"Threshold: {threshold:,} files",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="red")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Service account distribution using Plotly
                if 'service_accounts' in user_recs and user_recs['service_accounts']:
                    service_account_data = pd.DataFrame([
                        {
                            'Account': account['account_name'],
                            'File Count': account['total_files'],
                            'Folder Count': len(account['folders'])
                        }
                        for account in user_recs['service_accounts']
                    ])
                    
                    # Create a figure with subplots
                    fig = make_subplots(
                        rows=1, cols=2,
                        specs=[[{"type": "bar"}, {"type": "pie"}]],
                        subplot_titles=("File Count by Service Account", "Folder Distribution")
                    )
                    
                    # Add bar chart
                    fig.add_trace(
                        go.Bar(
                            x=service_account_data['Account'],
                            y=service_account_data['File Count'],
                            text=service_account_data['File Count'].apply(lambda x: f"{x:,}"),
                            name="File Count",
                            marker_color='rgb(55, 83, 109)'
                        ),
                        row=1, col=1
                    )
                    
                    # Add pie chart
                    fig.add_trace(
                        go.Pie(
                            labels=service_account_data['Account'],
                            values=service_account_data['Folder Count'],
                            name="Folder Count"
                        ),
                        row=1, col=2
                    )
                    
                    # Add threshold line to bar chart
                    threshold = st.session_state.service_account_threshold
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(service_account_data)-0.5,
                        y0=threshold,
                        y1=threshold,
                        line=dict(color="red", width=2, dash="dash"),
                        row=1, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title_text=f"Service Account Analysis for {selected_viz_user}",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Folder treemap visualization
                    if 'recommended_splits' in user_recs and user_recs['recommended_splits']:
                        # Prepare data for treemap
                        folder_data = []
                        for folder in user_recs['recommended_splits']:
                            # Split path into components
                            path_parts = folder['folder_path'].strip('/').split('/')
                            
                            # Create entries for each level of the path
                            current_path = ""
                            parent_path = ""
                            
                            for i, part in enumerate(path_parts):
                                current_path = current_path + "/" + part if current_path else part
                                
                                # Only add the leaf node with the file count
                                if i == len(path_parts) - 1:
                                    folder_data.append({
                                        'id': current_path,
                                        'parent': parent_path if parent_path else "",
                                        'name': part,
                                        'value': folder['recommended_files_to_move'],
                                        'assigned_to': folder.get('assigned_to', 'Not Assigned')
                                    })
                                else:
                                    folder_data.append({
                                        'id': current_path,
                                        'parent': parent_path if parent_path else "",
                                        'name': part,
                                        'value': None
                                    })
                                
                                parent_path = current_path
                        
                        # Create DataFrame
                        treemap_df = pd.DataFrame(folder_data)
                        
                        # Create treemap
                        fig = px.treemap(
                            treemap_df,
                            ids='id',
                            names='name',
                            parents='parent',
                            values='value',
                            color='assigned_to',
                            title=f'Folder Structure for {selected_viz_user}',
                            hover_data=['value'],
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        fig.update_traces(
                            textinfo="label+value",
                            hovertemplate='<b>%{label}</b><br>Files: %{value:,}<br>Assigned to: %{customdata[0]}'
                        )
                        
                        fig.update_layout(height=800)
                        
                        st.plotly_chart(fig, use_container_width=True)

def process_data(df: pd.DataFrame, threshold: int, progress_bar=None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """
    Process the data and generate recommendations.
    
    Args:
        df: DataFrame containing folder data
        threshold: Maximum number of files a user should have
        progress_bar: Streamlit progress bar
    
    Returns:
        Tuple containing:
        - DataFrame of users exceeding the threshold
        - Dictionary of recommendations for each user
        - Dictionary of visualizations for each user
        - Summary table as a DataFrame
        - Folder splits table as a DataFrame
    """
    try:
        # Create recommender
        recommender = FolderSplitRecommender(df, threshold)
        if progress_bar:
            progress_bar.progress(0.2)
        
        # Analyze data
        users_exceeding, recommendations, visualizations, summary_table, folder_splits_table = recommender.analyze()
        if progress_bar:
            progress_bar.progress(1.0)
        
        return users_exceeding, recommendations, visualizations, summary_table, folder_splits_table
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)  # This will show the full stack trace
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return None, None, None, None, None

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
    """)
    
    # Create sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Set threshold with user input
        threshold = st.number_input("File count threshold", min_value=100000, max_value=10000000, value=500000, step=50000)
        st.write(f"Using file count threshold: {threshold:,}")
        st.session_state.threshold = threshold
        
        # Add theme selector
        st.subheader("UI Settings")
        theme = st.selectbox("Theme", options=["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
    
    # Upload data
    st.header("Upload Data")
    
    # Show progress
    progress_bar = st.progress(0)
    
    try:
        # Use file uploader for deployment compatibility
        uploaded_file = st.file_uploader("Upload folder data CSV", type=["csv"])
        
        # Check if file is uploaded
        if uploaded_file is not None:
            # Start timing
            start_time = time.time()
            
            # Load data from uploaded file
            df = pd.read_csv(uploaded_file)
            progress_bar.progress(0.1)
            
            # Store threshold in session state
            st.session_state.threshold = threshold
            
            # Process data
            users_exceeding, recommendations, visualizations, summary_table, folder_splits_table = process_data(df, threshold, progress_bar)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            st.session_state.processing_time = processing_time
            
            # Store results in session state
            st.session_state.users_exceeding = users_exceeding
            st.session_state.recommendations = recommendations
            st.session_state.visualizations = visualizations
            st.session_state.summary_table = summary_table
            st.session_state.folder_splits_table = folder_splits_table
            
            # Mark analysis as complete
            st.session_state.analysis_complete = True
            
            # Display results
            display_results()
        else:
            st.info("Please upload a CSV file to begin analysis.")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
