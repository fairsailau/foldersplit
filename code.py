import streamlit as st
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
from typing import Dict, List, Any, Tuple, Optional, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state for preserving data between reruns
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None
if 'summary_table' not in st.session_state:
    st.session_state.summary_table = None
if 'users_exceeding' not in st.session_state:
    st.session_state.users_exceeding = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

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
        self.users_exceeding = None
        self.recommendations = {}
        
        # Validate input data
        self._validate_input_data()
        
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
        
    def calculate_user_stats(self) -> pd.DataFrame:
        """
        Calculate statistics for each user and identify those exceeding threshold.
        
        This method calculates the total file count per user by summing only level 1 folders,
        as file counts of level 2, 3, etc. are already included in level 1 counts.
        
        Returns:
            DataFrame of users exceeding the threshold with columns:
            - Owner: User email
            - total_file_count: Total number of files owned
            - total_size_mb: Total size in MB
            - folder_count: Number of level 1 folders
        """
        logger.info("Calculating user statistics...")
        st.write("Calculating user statistics...")
        
        try:
            # Filter to only level 1 folders and group by user
            level1_df = self.df[self.df['Level'] == 1]
            
            # Group by user and calculate total file count from level 1 folders only
            self.user_stats = level1_df.groupby('Owner').agg(
                total_file_count=('File Count', 'sum'),
                total_size_mb=('Size (MB)', 'sum'),
                folder_count=('Folder ID', 'count')
            ).sort_values('total_file_count', ascending=False).reset_index()
            
            # Identify users exceeding the threshold
            self.users_exceeding = self.user_stats[self.user_stats['total_file_count'] > self.file_threshold]
            
            logger.info(f"Found {len(self.users_exceeding)} users exceeding the threshold of {self.file_threshold:,} files")
            
            return self.users_exceeding
        
        except Exception as e:
            error_msg = f"Error calculating user statistics: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def identify_nested_folders(self) -> pd.DataFrame:
        """
        Identify parent-child relationships between folders and calculate direct file counts.
        
        This method calculates the "direct_file_count" for each folder, which represents the
        number of files directly in the folder (excluding subfolders). This is important for
        understanding the folder structure and making better split recommendations.
        
        Returns:
            DataFrame with added 'direct_file_count' column
        """
        logger.info("Identifying nested folder relationships...")
        st.write("Identifying nested folder relationships...")
        
        try:
            # Add path length for sorting (process shorter paths first)
            self.df['path_length'] = self.df['Path'].str.len()
            self.df = self.df.sort_values('path_length')
            
            # Initialize direct file count with total file count
            self.df['direct_file_count'] = self.df['File Count']
            
            # Create a dictionary for faster lookups
            path_to_idx = {row['Path']: i for i, row in self.df.iterrows()}
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            total_rows = len(self.df)
            
            progress_bar = st.progress(0)
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = self.df.iloc[start_idx:end_idx]
                
                for _, row in batch.iterrows():
                    path = row['Path']
                    # Skip root paths that don't have parent folders
                    if path.count('/') <= 1:
                        continue
                    
                    # Find parent folder
                    parent_path = '/'.join(path.split('/')[:-1]) + '/'
                    if parent_path in path_to_idx:
                        parent_idx = path_to_idx[parent_path]
                        # Subtract this folder's file count from parent's direct file count
                        self.df.at[parent_idx, 'direct_file_count'] -= row['File Count']
                
                # Update progress bar
                progress_bar.progress(min(end_idx / total_rows, 1.0))
            
            # Ensure direct file counts are not negative
            self.df['direct_file_count'] = self.df['direct_file_count'].clip(lower=0)
            
            logger.info("Nested folder relationships identified successfully")
            return self.df
            
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
        for selected_folder in selected_folders:
            # Check if folder_path starts with selected_folder
            # We need to ensure it's a proper subfolder by checking for the trailing slash
            if folder_path.startswith(selected_folder) and folder_path != selected_folder:
                return True
        return False
    
    def assign_to_service_accounts(self, user_email: str, folders_to_split: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign folders to service accounts, keeping related folders together when possible
        and ensuring each service account stays under the threshold.
        
        This method groups folders by common parent paths to keep related folders together,
        then assigns them to service accounts while ensuring no account exceeds the threshold.
        
        Args:
            user_email: Email of the user whose folders are being split
            folders_to_split: List of folder dictionaries to assign to service accounts
            
        Returns:
            List of service account dictionaries, each containing:
            - account_name: Name of the service account (e.g., "service_account_1")
            - folders: List of folders assigned to this account
            - total_files: Total number of files assigned to this account
        """
        logger.info(f"Assigning folders to service accounts for user: {user_email}")
        st.write(f"Assigning folders to service accounts for user: {user_email}")
        
        try:
            # If no folders to split, return empty list instead of creating empty service accounts
            if not folders_to_split:
                logger.info(f"No folders to split for user {user_email}")
                return []
                
            # Sort folders by path to keep related folders together
            folders_to_split.sort(key=lambda x: x['folder_path'])
            
            service_accounts = []
            current_account = {
                'account_name': f'service_account_1',
                'folders': [],
                'total_files': 0
            }
            
            # Group folders by common parent paths (first level)
            folder_groups = {}
            for folder in folders_to_split:
                # Extract first level path component
                path_parts = folder['folder_path'].strip('/').split('/')
                if len(path_parts) > 0:
                    group_key = path_parts[0]
                    if group_key not in folder_groups:
                        folder_groups[group_key] = []
                    folder_groups[group_key].append(folder)
                else:
                    # Handle root folders
                    if 'root' not in folder_groups:
                        folder_groups['root'] = []
                    folder_groups['root'].append(folder)
            
            # Process each group of related folders
            for group_key, group_folders in folder_groups.items():
                # Sort folders within group by file count (descending)
                group_folders.sort(key=lambda x: x['current_file_count'], reverse=True)
                
                for folder in group_folders:
                    # Check if adding this folder would exceed the threshold for current account
                    if current_account['total_files'] + folder['recommended_files_to_move'] > self.file_threshold:
                        # Only add the current account to service_accounts if it has folders
                        if current_account['folders']:
                            service_accounts.append(current_account)
                        
                        # Create a new account
                        current_account = {
                            'account_name': f'service_account_{len(service_accounts) + 1}',
                            'folders': [],
                            'total_files': 0
                        }
                    
                    # Add folder to current service account
                    folder['assigned_to'] = current_account['account_name']
                    current_account['folders'].append(folder)
                    current_account['total_files'] += folder['recommended_files_to_move']
            
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
        
        This method prioritizes folders with the highest file count that are below the threshold,
        while respecting parent-child relationships. It continues adding folders until
        the user's total file count is reduced to the threshold or less, then assigns
        the selected folders to service accounts.
        
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
                    'recommended_splits': []
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
                    
                    # Skip folders that exceed the threshold instead of doing partial splits
                    if folder_file_count > self.file_threshold:
                        logger.info(f"Skipping large folder: {folder_path} ({folder_file_count:,} files) - exceeds threshold")
                        st.write(f"Skipping large folder: {folder_path} ({folder_file_count:,} files) - exceeds threshold")
                        continue
                    
                    # Check if this folder is a good candidate (≤ threshold files)
                    # Calculate how much this split would reduce the total
                    new_total = remaining_files - folder_file_count
                    
                    # Add to candidates
                    candidates.append({
                        'folder_path': folder_path,
                        'folder_name': folder['Folder Name'],
                        'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                        'level': int(folder['Level']),
                        'current_file_count': int(folder_file_count),
                        'direct_file_count': int(folder['direct_file_count']),
                        'new_total': new_total
                    })
                
                # Sort candidates by file count (descending) to prioritize larger folders
                candidates.sort(key=lambda x: x['current_file_count'], reverse=True)
                
                logger.info(f"Found {len(candidates)} candidates across all levels")
                st.write(f"Found {len(candidates)} candidates across all levels")
                
                # Add candidates to recommendations until we're below threshold or no more candidates
                for candidate in candidates:
                    # Skip if we're already below threshold
                    if remaining_files <= self.file_threshold:
                        break
                    
                    # Add this candidate to recommendations
                    recommendations['recommended_splits'].append({
                        'folder_path': candidate['folder_path'],
                        'folder_name': candidate['folder_name'],
                        'folder_id': candidate['folder_id'],
                        'level': candidate['level'],
                        'current_file_count': candidate['current_file_count'],
                        'direct_file_count': candidate['direct_file_count'],
                        'recommended_files_to_move': candidate['current_file_count'],
                        'new_total_after_split': candidate['new_total']
                    })
                    
                    # Add this folder to the selected folders list
                    selected_folder_paths.append(candidate['folder_path'])
                    
                    # Update remaining files
                    remaining_files = candidate['new_total']
                    logger.info(f"Added folder: {candidate['folder_path']} (Level {candidate['level']}), file count: {candidate['current_file_count']:,}, new total: {remaining_files:,}")
                    st.write(f"  Added folder: {candidate['folder_path']} (Level {candidate['level']}), file count: {candidate['current_file_count']:,}, new total: {remaining_files:,}")
                
                st.write(f"After processing all candidates, remaining files: {remaining_files:,}")
                
                # Calculate total recommended moves and remaining excess
                total_recommended_moves = sum([rec.get('recommended_files_to_move', 0) for rec in recommendations['recommended_splits']])
                
                # Ensure total_recommended_moves doesn't exceed the original file count
                total_recommended_moves = min(total_recommended_moves, total_file_count)
                
                recommendations['total_recommended_moves'] = total_recommended_moves
                recommendations['remaining_excess_files'] = int(remaining_files - self.file_threshold) if remaining_files > self.file_threshold else 0
                
                # Ensure final_file_count is correctly set to the actual remaining files
                recommendations['final_file_count'] = int(remaining_files)
                
                # Assign folders to service accounts
                service_accounts = self.assign_to_service_accounts(user_email, recommendations['recommended_splits'])
                recommendations['service_accounts'] = service_accounts
                
                # Save recommendations for this user
                self.recommendations[user_email] = recommendations
                
                logger.info(f"Final recommendations for {user_email}:")
                logger.info(f"Total recommended moves: {total_recommended_moves:,}")
                logger.info(f"Final file count after all splits: {recommendations['final_file_count']:,}")
                logger.info(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
                logger.info(f"Number of service accounts needed: {len(service_accounts)}")
                
                st.write(f"Final recommendations for {user_email}:")
                st.write(f"Total recommended moves: {total_recommended_moves:,}")
                st.write(f"Final file count after all splits: {recommendations['final_file_count']:,}")
                st.write(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
                st.write(f"Number of service accounts needed: {len(service_accounts)}")
            
            return self.recommendations
            
        except Exception as e:
            error_msg = f"Error prioritizing folders: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def visualize_recommendations(self) -> Dict[str, Dict[str, plt.Figure]]:
        """
        Create visualizations of the recommendations.
        
        This method generates various visualizations for each user:
        - Recommended files to move by folder
        - Current vs. recommended file count
        - Before and after total file count
        - Distribution across service accounts
        - Folder distribution across service accounts
        - Folder size distribution (NEW)
        - Folder level distribution (NEW)
        
        Returns:
            Dictionary of visualizations for each user
        """
        logger.info("Creating visualizations...")
        st.write("Creating visualizations...")
        
        try:
            visualizations = {}
            
            # For each user with recommendations
            for user_email, user_recs in self.recommendations.items():
                # Skip if no recommended splits
                if len(user_recs['recommended_splits']) == 0:
                    continue
                    
                # Create a DataFrame from the recommended splits
                splits_df = pd.DataFrame(user_recs['recommended_splits'])
                
                # Add a column to indicate partial splits
                splits_df['split_type'] = splits_df.apply(
                    lambda x: 'Partial Split' if x.get('is_partial_split', False) else 'Complete Split',
                    axis=1
                )
                
                user_visualizations = {}
                
                # 1. Plot recommended files to move by folder
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(splits_df['folder_name'], splits_df['recommended_files_to_move'], 
                        color=splits_df['split_type'].map({'Complete Split': 'green', 'Partial Split': 'orange'}))
                
                # Add data labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                            f'{int(width):,}',
                            ha='left', va='center')
                
                ax.set_xlabel('Recommended Files to Move')
                ax.set_ylabel('Folder Name')
                ax.set_title(f'Recommended Folder Splits for {user_email}')
                ax.legend(handles=[
                    plt.Rectangle((0,0),1,1, color='green', label='Complete Split'),
                    plt.Rectangle((0,0),1,1, color='orange', label='Partial Split')
                ])
                plt.tight_layout()
                user_visualizations['recommendations'] = fig
                
                # 2. Plot current vs. recommended file count
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Sort by file count (descending)
                splits_df_sorted = splits_df.sort_values('current_file_count', ascending=False)
                
                x = range(len(splits_df_sorted))
                width = 0.35
                
                ax.bar(x, splits_df_sorted['current_file_count'], width, label='Current File Count')
                ax.bar(x, splits_df_sorted['recommended_files_to_move'], width, 
                       label='Files to Move', alpha=0.7, color='red')
                
                ax.axhline(y=self.file_threshold, color='green', linestyle='--', 
                           label=f'Threshold ({self.file_threshold:,} files)')
                
                ax.set_xlabel('Folder Name')
                ax.set_ylabel('File Count')
                ax.set_title(f'Current vs. Recommended File Count for {user_email}')
                ax.set_xticks(x)
                ax.set_xticklabels(splits_df_sorted['folder_name'], rotation=90)
                ax.legend()
                plt.tight_layout()
                user_visualizations['current_vs_recommended'] = fig
                
                # 3. Plot before and after total file count
                fig, ax = plt.subplots(figsize=(10, 6))
                
                labels = ['Before Split', 'After All Splits']
                values = [user_recs['total_file_count'], user_recs['final_file_count']]
                colors = ['#ff9999', '#66b3ff']
                
                ax.bar(labels, values, color=colors)
                ax.axhline(y=self.file_threshold, color='green', linestyle='--', 
                           label=f'Threshold ({self.file_threshold:,} files)')
                
                # Add data labels
                for i, v in enumerate(values):
                    ax.text(i, v + v*0.02, f'{int(v):,}', ha='center')
                
                ax.set_ylabel('Total File Count')
                ax.set_title(f'Before vs. After All Splits for {user_email}')
                ax.legend()
                plt.tight_layout()
                user_visualizations['before_after'] = fig
                
                # 4. Plot distribution across service accounts
                if 'service_accounts' in user_recs and user_recs['service_accounts']:
                    # Create data for service account distribution
                    account_names = [account['account_name'] for account in user_recs['service_accounts']]
                    account_names.insert(0, f"{user_email} (After Split)")
                    
                    file_counts = [account['total_files'] for account in user_recs['service_accounts']]
                    file_counts.insert(0, user_recs['final_file_count'])
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.bar(account_names, file_counts, color=['#66b3ff'] + ['#ff9999'] * len(user_recs['service_accounts']))
                    
                    # Add threshold line
                    ax.axhline(y=self.file_threshold, color='green', linestyle='--', 
                               label=f'Threshold ({self.file_threshold:,} files)')
                    
                    # Add data labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                                f'{int(height):,}', ha='center', va='bottom')
                    
                    ax.set_xlabel('Account')
                    ax.set_ylabel('File Count')
                    ax.set_title(f'File Distribution Across Accounts for {user_email}')
                    plt.xticks(rotation=45, ha='right')
                    ax.legend()
                    plt.tight_layout()
                    user_visualizations['service_account_distribution'] = fig
                    
                    # 5. Create a visualization showing which folders go to which service account
                    # Group folders by service account
                    account_folders = {}
                    for folder in user_recs['recommended_splits']:
                        account = folder.get('assigned_to', 'Unknown')
                        if account not in account_folders:
                            account_folders[account] = []
                        account_folders[account].append(folder)
                    
                    # Create a stacked bar chart showing folder distribution
                    account_labels = list(account_folders.keys())
                    folder_counts = []
                    folder_labels = []
                    
                    # Prepare data for stacked bar chart
                    for account in account_labels:
                        folders = account_folders[account]
                        for folder in folders:
                            folder_counts.append(folder['recommended_files_to_move'])
                            folder_labels.append(f"{folder['folder_name']} ({account})")
                    
                    # Create horizontal stacked bar chart
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Use a colormap to assign different colors to different folders
                    colors = plt.cm.viridis(np.linspace(0, 1, len(folder_counts)))
                    
                    # Create the stacked bar chart
                    y_pos = 0
                    for i, (count, label, color) in enumerate(zip(folder_counts, folder_labels, colors)):
                        ax.barh(y_pos, count, color=color, label=label)
                        # Add text label inside the bar if there's enough space
                        if count > max(folder_counts) * 0.05:
                            ax.text(count/2, y_pos, label, ha='center', va='center', color='white')
                        y_pos += 1
                    
                    ax.set_yticks([])  # Hide y-axis ticks
                    ax.set_xlabel('File Count')
                    ax.set_title(f'Folder Distribution Across Service Accounts for {user_email}')
                    
                    # Create a custom legend
                    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
                    ax.legend(handles, folder_labels, loc='upper right', bbox_to_anchor=(1.1, 1), 
                              ncol=1, fontsize='small')
                    
                    plt.tight_layout()
                    user_visualizations['folder_distribution'] = fig
                    
                    # NEW: 6. Create a pie chart showing folder size distribution
                    fig, ax = plt.subplots(figsize=(10, 10))
                    
                    # Get folder sizes if available in the DataFrame
                    if 'Size (MB)' in self.df.columns:
                        # Get folders being moved
                        moved_folder_paths = [folder['folder_path'] for folder in user_recs['recommended_splits']]
                        moved_folders_df = self.df[self.df['Path'].isin(moved_folder_paths)]
                        
                        # Group by assigned service account
                        size_by_account = {}
                        for folder in user_recs['recommended_splits']:
                            account = folder.get('assigned_to', 'Unknown')
                            folder_path = folder['folder_path']
                            folder_size = moved_folders_df[moved_folders_df['Path'] == folder_path]['Size (MB)'].values
                            
                            if len(folder_size) > 0:
                                if account not in size_by_account:
                                    size_by_account[account] = 0
                                size_by_account[account] += folder_size[0]
                        
                        # Create pie chart
                        labels = list(size_by_account.keys())
                        sizes = list(size_by_account.values())
                        
                        if sizes:  # Only create pie chart if we have size data
                            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                                   colors=plt.cm.tab20.colors[:len(sizes)])
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            ax.set_title(f'Folder Size Distribution Across Service Accounts for {user_email}')
                            user_visualizations['size_distribution'] = fig
                    
                    # NEW: 7. Create a bar chart showing folder level distribution
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Count folders by level for each service account
                    level_counts = {}
                    for account in account_labels:
                        folders = account_folders[account]
                        level_counts[account] = {}
                        for folder in folders:
                            level = folder['level']
                            if level not in level_counts[account]:
                                level_counts[account][level] = 0
                            level_counts[account][level] += 1
                    
                    # Prepare data for grouped bar chart
                    all_levels = sorted(set(level for account_levels in level_counts.values() 
                                           for level in account_levels.keys()))
                    
                    x = np.arange(len(all_levels))
                    width = 0.8 / len(level_counts)
                    
                    # Plot bars for each account
                    for i, (account, levels) in enumerate(level_counts.items()):
                        counts = [levels.get(level, 0) for level in all_levels]
                        ax.bar(x + i*width - 0.4 + width/2, counts, width, label=account)
                    
                    ax.set_xlabel('Folder Level')
                    ax.set_ylabel('Number of Folders')
                    ax.set_title(f'Folder Level Distribution Across Service Accounts for {user_email}')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'Level {level}' for level in all_levels])
                    ax.legend()
                    
                    plt.tight_layout()
                    user_visualizations['level_distribution'] = fig
                
                visualizations[user_email] = user_visualizations
            
            return visualizations
            
        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise

    def get_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of recommendations for all users.
        
        Returns:
            DataFrame with summary information for all users and service accounts
        """
        try:
            summary_data = []
            
            # First add original users with their updated file counts
            for user_email, user_recs in self.recommendations.items():
                # Use the actual final file count after splitting
                final_count = user_recs['final_file_count']
                
                # Calculate files to move
                files_to_move = user_recs['total_recommended_moves']
                
                summary_data.append({
                    'User': user_email,
                    'Before Split': user_recs['total_file_count'],
                    'After All Splits': final_count,
                    'Files to Move': files_to_move,
                    'Service Accounts': len(user_recs.get('service_accounts', [])),
                    'Status': 'Success' if final_count <= self.file_threshold else 'Partial Success'
                })
                
                # Then add service accounts
                if 'service_accounts' in user_recs:
                    for account in user_recs['service_accounts']:
                        summary_data.append({
                            'User': account['account_name'],
                            'Before Split': 0,  # Service accounts start with 0 files
                            'After All Splits': account['total_files'],
                            'Files to Move': account['total_files'],  # All files are moved to this account
                            'Service Accounts': '',  # Service accounts don't have their own service accounts
                            'Status': 'Success' if account['total_files'] <= self.file_threshold else 'Partial Success'
                        })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            error_msg = f"Error creating summary table: {str(e)}"
            logger.error(error_msg)
            raise

# Main function to run the Streamlit application
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide")  # Set page to wide mode for better display
    
    st.title("Folder Splitting Recommendation Tool (Improved)")
    
    st.write("""
    This tool analyzes folder ownership data and provides recommendations for splitting content 
    based on file count thresholds. It identifies users who own more than 500,000 files and 
    recommends which folders to split to bring users below this threshold.
    
    **Improvements in this version:**
    1. Skips large folders over 500,000 files instead of attempting partial splits
    2. Ensures no empty service accounts are created
    3. Focuses on subfolders for better splitting options
    """)
    
    # About this tool section
    st.subheader("About This Tool")
    st.write("""
    This tool analyzes folder ownership data and provides recommendations for splitting content 
    based on file count thresholds.
    
    **Key Features:**
    1. Correctly calculates total file counts per user
    2. Identifies users exceeding the threshold
    3. Recommends which folders to split to service accounts
    4. Ensures no service account exceeds the threshold
    5. Keeps related folders together when possible
    6. Provides visualizations of the recommendations
    """)
    
    # File upload section
    st.subheader("Upload Data")
    
    # Create tabs for upload and sample data
    upload_tabs = st.tabs(["Upload CSV", "Use Sample Data"])
    
    with upload_tabs[0]:
        uploaded_file = st.file_uploader("Upload a CSV file with folder data", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                st.write("Data uploaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    with upload_tabs[1]:
        st.write("Use the sample data provided with the tool.")
        if st.button("Load Sample Data"):
            try:
                # Load sample data from the current directory
                df = pd.read_csv("folder_data.csv")
                st.session_state.uploaded_data = df
                st.write("Sample data loaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Threshold setting
    st.subheader("Set Threshold")
    threshold = st.number_input(
        "Maximum number of files per user",
        min_value=100000,
        max_value=10000000,
        value=500000,
        step=100000,
        help="Users with more files than this threshold will be identified for folder splitting."
    )
    st.session_state.threshold = threshold
    
    # Analysis section
    st.subheader("Analysis")
    
    # Check if data is uploaded
    if 'uploaded_data' in st.session_state:
        if st.button("Generate Recommendations"):
            # Run the analysis
            users_exceeding, recommendations, visualizations, summary_table = generate_recommendations()
            
            # Display results if analysis is complete
            if st.session_state.analysis_complete:
                display_results()
    else:
        st.info("Please upload data or load sample data to continue.")
    
    # Display results if analysis is already complete
    if st.session_state.analysis_complete:
        display_results()

def generate_recommendations():
    """Generate recommendations based on the uploaded data and threshold."""
    try:
        st.subheader("Analysis Results")
        
        # Create progress bar for overall process
        progress_bar = st.progress(0)
        
        # Get the uploaded data and threshold from session state
        df = st.session_state.uploaded_data
        threshold = st.session_state.threshold
        
        # Create recommender
        recommender = FolderSplitRecommender(df, threshold)
        progress_bar.progress(0.1)
        
        # Calculate user statistics
        users_exceeding = recommender.calculate_user_stats()
        progress_bar.progress(0.2)
        
        # Store users_exceeding in session state
        st.session_state.users_exceeding = users_exceeding
        
        # Display users exceeding threshold
        st.write(f"Found {len(users_exceeding)} users exceeding the threshold of {threshold:,} files:")
        st.dataframe(users_exceeding)
        
        # Generate recommendations
        with st.spinner("Generating recommendations... This may take a few minutes for large datasets."):
            recommendations = recommender.prioritize_folders()
        progress_bar.progress(0.6)
        
        # Store recommendations in session state
        st.session_state.recommendations = recommendations
        
        # Create visualizations
        with st.spinner("Creating visualizations..."):
            visualizations = recommender.visualize_recommendations()
        progress_bar.progress(0.8)
        
        # Store visualizations in session state
        st.session_state.visualizations = visualizations
        
        # Get summary table
        summary_table = recommender.get_summary_table()
        progress_bar.progress(0.9)
        
        # Store summary table in session state
        st.session_state.summary_table = summary_table
        
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

def display_results():
    """Display the results of the analysis."""
    # Get the results from session state
    users_exceeding = st.session_state.users_exceeding
    recommendations = st.session_state.recommendations
    visualizations = st.session_state.visualizations
    summary_table = st.session_state.summary_table
    
    if not all([users_exceeding is not None, recommendations, visualizations, summary_table is not None]):
        st.warning("No analysis results to display. Please generate recommendations first.")
        return
    
    # Display recommendations for each user
    st.subheader("Recommendations by User")
    
    # Display summary table
    st.dataframe(summary_table)
    
    # Download CSV button
    csv = summary_table.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="summary.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
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
                    # Display service account distribution visualization
                    if 'service_account_distribution' in user_viz:
                        st.pyplot(user_viz['service_account_distribution'])
                    
                    # Display folder distribution visualization
                    if 'folder_distribution' in user_viz:
                        st.pyplot(user_viz['folder_distribution'])
                
                with viz_tabs[2]:
                    # Display additional visualizations
                    if 'size_distribution' in user_viz:
                        st.pyplot(user_viz['size_distribution'])
                    
                    if 'level_distribution' in user_viz:
                        st.pyplot(user_viz['level_distribution'])
            else:
                st.write("No suitable folders found for splitting.")
        
        # Create a downloadable ZIP with all results
        st.subheader("Download All Results")
        
        # Create a buffer for the ZIP file
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w') as zip_file:
            # Add summary table
            zip_file.writestr('summary_table.csv', summary_table.to_csv(index=False))
            
            # Add recommendations JSON
            zip_file.writestr('recommendations.json', json.dumps(recommendations, default=str, indent=4))
            
            # Add service account assignments
            service_account_data = []
            for user_email, user_recs in recommendations.items():
                if 'service_accounts' in user_recs and user_recs['service_accounts']:
                    for account in user_recs['service_accounts']:
                        for folder in account['folders']:
                            service_account_data.append({
                                'User': user_email,
                                'Service Account': account['account_name'],
                                'Folder Name': folder['folder_name'],
                                'Folder Path': folder['folder_path'],
                                'Files to Move': folder['recommended_files_to_move'],
                                'Split Type': 'Partial Split' if folder.get('is_partial_split', False) else 'Complete Split',
                                'Level': folder['level']
                            })
            
            if service_account_data:
                service_account_df = pd.DataFrame(service_account_data)
                zip_file.writestr('service_account_assignments.csv', service_account_df.to_csv(index=False))
            
            # Add README file with explanation
            readme_content = """# Folder Split Recommendations

This ZIP file contains the results of the Folder Split Recommendation Tool analysis.

## Files Included:

1. **summary_table.csv**: Overview of all users and service accounts with file counts before and after splits.
2. **recommendations.json**: Detailed recommendations in JSON format.
3. **service_account_assignments.csv**: Detailed mapping of folders to service accounts.

## How to Use These Results:

The recommendations suggest moving specific folders from users who exceed the threshold to service accounts.
Each service account is kept under the threshold to ensure optimal performance.

For implementation, follow these steps:
1. Review the summary table to understand the overall impact
2. Check the service account assignments to see which folders should be moved
3. Implement the moves according to the recommendations

For questions or support, please contact your system administrator.
"""
            zip_file.writestr('README.md', readme_content)
        
        # Create download link for ZIP
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.read()).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="folder_split_recommendations.zip">Download All Results (ZIP)</a>'
        st.markdown(href, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
