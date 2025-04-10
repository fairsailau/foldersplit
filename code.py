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
        Assign folders to service accounts using a best-fit bin packing algorithm,
        ensuring each service account stays under the threshold while minimizing
        the total number of service accounts needed.
        
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
            # Sort all folders by file count (descending) to prioritize larger folders
            folders_to_split.sort(key=lambda x: x['recommended_files_to_move'], reverse=True)
            
            service_accounts = []
            
            # Process each folder
            for folder in folders_to_split:
                # Try to find the best existing service account that can fit this folder
                best_fit_account = None
                min_remaining_space = self.file_threshold + 1  # Initialize with a value larger than threshold
                
                for account in service_accounts:
                    remaining_space = self.file_threshold - account['total_files']
                    
                    # If this account can fit the folder and has less remaining space than current best
                    if folder['recommended_files_to_move'] <= remaining_space and remaining_space < min_remaining_space:
                        best_fit_account = account
                        min_remaining_space = remaining_space
                
                # If no suitable account found, create a new one
                if best_fit_account is None:
                    best_fit_account = {
                        'account_name': f'service_account_{len(service_accounts) + 1}',
                        'folders': [],
                        'total_files': 0
                    }
                    service_accounts.append(best_fit_account)
                
                # Add folder to the selected service account
                folder['assigned_to'] = best_fit_account['account_name']
                best_fit_account['folders'].append(folder)
                best_fit_account['total_files'] += folder['recommended_files_to_move']
            
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
        
        This method prioritizes folders with the highest file count regardless of level,
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
                    
                    # Check if this folder is a good candidate (≤ threshold files)
                    if folder_file_count <= self.file_threshold:
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
                
                # If we still haven't reached the threshold, look for partial splits
                if remaining_files > self.file_threshold:
                    logger.info("Still above threshold after all candidates, looking for partial splits...")
                    st.write("Still above threshold after all candidates, looking for partial splits...")
                    
                    # Get all folders sorted by file count (descending)
                    all_folders = user_folders.sort_values('File Count', ascending=False)
                    
                    for _, folder in all_folders.iterrows():
                        folder_path = folder['Path']
                        folder_file_count = folder['File Count']
                        
                        # Skip folders that are too small
                        if folder_file_count < 10000:
                            continue
                        
                        # Skip if this folder is a subfolder of any already selected folder
                        if self._is_subfolder_of_any(folder_path, selected_folder_paths):
                            continue
                        
                        # For larger folders, calculate how many files to move
                        files_to_move = remaining_files - self.file_threshold
                        
                        # Only recommend if this folder has enough files
                        if files_to_move > 0 and files_to_move < folder_file_count:
                            recommendations['recommended_splits'].append({
                                'folder_path': folder_path,
                                'folder_name': folder['Folder Name'],
                                'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                                'level': int(folder['Level']),
                                'current_file_count': int(folder_file_count),
                                'direct_file_count': int(folder['direct_file_count']),
                                'recommended_files_to_move': int(files_to_move),
                                'new_total_after_split': self.file_threshold,
                                'is_partial_split': True
                            })
                            
                            # Add this folder to the selected folders list
                            selected_folder_paths.append(folder_path)
                            
                            # Update remaining files
                            remaining_files = self.file_threshold
                            logger.info(f"Added partial split of folder: {folder_path} (Level {int(folder['Level'])}), files to move: {files_to_move:,}, new total: {remaining_files:,}")
                            st.write(f"Added partial split of folder: {folder_path} (Level {int(folder['Level'])}), files to move: {files_to_move:,}, new total: {remaining_files:,}")
                            break
                
                # Calculate total recommended moves and remaining excess
                total_recommended_moves = sum([rec.get('recommended_files_to_move', 0) for rec in recommendations['recommended_splits']])
                
                # Ensure total_recommended_moves doesn't exceed the original file count
                total_recommended_moves = min(total_recommended_moves, total_file_count)
                
                recommendations['total_recommended_moves'] = total_recommended_moves
                recommendations['remaining_excess_files'] = int(remaining_files - self.file_threshold) if remaining_files > self.file_threshold else 0
                
                # Ensure final_file_count is correctly set to threshold or less
                recommendations['final_file_count'] = min(int(remaining_files), self.file_threshold)
                
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
                
                # 6. Folder size distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create size bins
                size_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
                size_labels = ['<10MB', '10-50MB', '50-100MB', '100-500MB', '500MB-1GB', '>1GB']
                
                # Count folders in each bin
                size_counts = []
                for i in range(len(size_bins)-1):
                    count = len(splits_df[(splits_df['Size (MB)'] >= size_bins[i]) & 
                                         (splits_df['Size (MB)'] < size_bins[i+1])])
                    size_counts.append(count)
                
                ax.bar(size_labels, size_counts, color='skyblue')
                ax.set_xlabel('Folder Size')
                ax.set_ylabel('Number of Folders')
                ax.set_title(f'Size Distribution of Recommended Folders for {user_email}')
                plt.tight_layout()
                user_visualizations['size_distribution'] = fig
                
                # 7. Folder level distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Count folders at each level
                level_counts = splits_df['level'].value_counts().sort_index()
                
                ax.bar(level_counts.index.astype(str), level_counts.values, color='lightgreen')
                ax.set_xlabel('Folder Level')
                ax.set_ylabel('Number of Folders')
                ax.set_title(f'Level Distribution of Recommended Folders for {user_email}')
                plt.tight_layout()
                user_visualizations['level_distribution'] = fig
                
                # Save visualizations for this user
                visualizations[user_email] = user_visualizations
            
            return visualizations
            
        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of recommendations for all users.
        
        Returns:
            DataFrame with summary information for each user
        """
        logger.info("Creating summary table...")
        
        try:
            summary_data = []
            
            for user_email, user_recs in self.recommendations.items():
                # Skip if no recommended splits
                if len(user_recs['recommended_splits']) == 0:
                    continue
                
                # Count folders at each level
                splits_df = pd.DataFrame(user_recs['recommended_splits'])
                level_counts = splits_df['level'].value_counts().to_dict()
                
                # Format level counts as string
                level_str = ', '.join([f"L{level}: {count}" for level, count in sorted(level_counts.items())])
                
                # Count partial splits
                partial_splits = sum(1 for split in user_recs['recommended_splits'] if split.get('is_partial_split', False))
                
                # Add row to summary data
                summary_data.append({
                    'User': user_email,
                    'Original File Count': user_recs['total_file_count'],
                    'Final File Count': user_recs['final_file_count'],
                    'Files Moved': user_recs['total_recommended_moves'],
                    'Folders Split': len(user_recs['recommended_splits']),
                    'Folder Levels': level_str,
                    'Partial Splits': partial_splits,
                    'Service Accounts': len(user_recs['service_accounts']) if 'service_accounts' in user_recs else 0,
                    'Remaining Excess': user_recs['remaining_excess_files']
                })
            
            # Create DataFrame from summary data
            summary_df = pd.DataFrame(summary_data)
            
            return summary_df
            
        except Exception as e:
            error_msg = f"Error creating summary table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def export_recommendations(self) -> Dict[str, Any]:
        """
        Export recommendations to various formats.
        
        Returns:
            Dictionary containing export data in different formats
        """
        logger.info("Exporting recommendations...")
        
        try:
            exports = {}
            
            # Export to JSON
            json_data = {}
            for user_email, user_recs in self.recommendations.items():
                # Create a simplified version for JSON export
                json_data[user_email] = {
                    'user_email': user_email,
                    'total_file_count': user_recs['total_file_count'],
                    'final_file_count': user_recs['final_file_count'],
                    'total_recommended_moves': user_recs['total_recommended_moves'],
                    'remaining_excess_files': user_recs['remaining_excess_files'],
                    'recommended_splits': []
                }
                
                # Add recommended splits
                for split in user_recs['recommended_splits']:
                    json_data[user_email]['recommended_splits'].append({
                        'folder_path': split['folder_path'],
                        'folder_name': split['folder_name'],
                        'folder_id': split['folder_id'],
                        'level': split['level'],
                        'current_file_count': split['current_file_count'],
                        'recommended_files_to_move': split['recommended_files_to_move'],
                        'is_partial_split': split.get('is_partial_split', False),
                        'assigned_to': split.get('assigned_to', '')
                    })
                
                # Add service accounts
                if 'service_accounts' in user_recs:
                    json_data[user_email]['service_accounts'] = []
                    for account in user_recs['service_accounts']:
                        json_data[user_email]['service_accounts'].append({
                            'account_name': account['account_name'],
                            'total_files': account['total_files'],
                            'folder_count': len(account['folders'])
                        })
            
            exports['json'] = json.dumps(json_data, indent=2)
            
            # Export to CSV
            csv_data = []
            for user_email, user_recs in self.recommendations.items():
                for split in user_recs['recommended_splits']:
                    csv_data.append({
                        'User': user_email,
                        'Folder Path': split['folder_path'],
                        'Folder Name': split['folder_name'],
                        'Folder ID': split['folder_id'],
                        'Level': split['level'],
                        'Current File Count': split['current_file_count'],
                        'Files to Move': split['recommended_files_to_move'],
                        'Partial Split': 'Yes' if split.get('is_partial_split', False) else 'No',
                        'Assigned To': split.get('assigned_to', '')
                    })
            
            csv_df = pd.DataFrame(csv_data)
            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            exports['csv'] = csv_buffer.getvalue()
            
            # Export to Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_df = self.create_summary_table()
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # User sheets
                for user_email, user_recs in self.recommendations.items():
                    # Create user sheet
                    user_df = pd.DataFrame(user_recs['recommended_splits'])
                    
                    # Select and rename columns for Excel
                    user_df = user_df[[
                        'folder_path', 'folder_name', 'folder_id', 'level', 
                        'current_file_count', 'recommended_files_to_move'
                    ]]
                    user_df.columns = [
                        'Folder Path', 'Folder Name', 'Folder ID', 'Level',
                        'Current File Count', 'Files to Move'
                    ]
                    
                    # Add partial split column
                    user_df['Partial Split'] = [
                        'Yes' if split.get('is_partial_split', False) else 'No'
                        for split in user_recs['recommended_splits']
                    ]
                    
                    # Add assigned to column
                    user_df['Assigned To'] = [
                        split.get('assigned_to', '')
                        for split in user_recs['recommended_splits']
                    ]
                    
                    # Write to Excel
                    sheet_name = user_email.split('@')[0][:31]  # Excel sheet names limited to 31 chars
                    user_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            excel_data = excel_buffer.getvalue()
            exports['excel'] = base64.b64encode(excel_data).decode('utf-8')
            
            # Export to ZIP (containing all formats)
            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                # Add JSON file
                zip_file.writestr('recommendations.json', exports['json'])
                
                # Add CSV file
                zip_file.writestr('recommendations.csv', exports['csv'])
                
                # Add Excel file
                zip_file.writestr('recommendations.xlsx', base64.b64decode(exports['excel']))
                
                # Add README file
                readme_text = """# Folder Split Recommendations

This ZIP file contains recommendations for splitting folders to reduce user file counts below the threshold.

## Files Included

- `recommendations.json`: JSON format of all recommendations
- `recommendations.csv`: CSV format of all folder splits
- `recommendations.xlsx`: Excel workbook with summary and per-user sheets

## Implementation Notes

1. Start with the highest priority folders (largest file counts)
2. For partial splits, move only the recommended number of files
3. Assign folders to service accounts as specified in the recommendations
"""
                zip_file.writestr('README.md', readme_text)
            
            zip_data = zip_buffer.getvalue()
            exports['zip'] = base64.b64encode(zip_data).decode('utf-8')
            
            return exports
            
        except Exception as e:
            error_msg = f"Error exporting recommendations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise

# Main Streamlit app
def main():
    st.title("Folder Split Recommender")
    
    st.write("""
    This tool analyzes folder ownership data and provides recommendations for splitting content
    based on file count thresholds. It identifies users who exceed the threshold and suggests
    which folders to split to bring them below the threshold.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload folder data CSV", type=['csv'])
    
    # Threshold input
    file_threshold = st.number_input(
        "File count threshold per user",
        min_value=100000,
        max_value=10000000,
        value=500000,
        step=100000,
        help="Maximum number of files a user should have"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.write(f"Loaded data with {len(df)} rows")
            
            # Show sample of the data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Create recommender
            recommender = FolderSplitRecommender(df, file_threshold)
            
            # Calculate user statistics
            if st.button("Analyze Data"):
                # Calculate user statistics
                users_exceeding = recommender.calculate_user_stats()
                st.session_state.users_exceeding = users_exceeding
                
                # Show users exceeding threshold
                st.subheader("Users Exceeding Threshold")
                st.dataframe(users_exceeding)
                
                # Generate recommendations if users are exceeding threshold
                if len(users_exceeding) > 0:
                    st.write(f"Found {len(users_exceeding)} users exceeding the threshold of {file_threshold:,} files")
                    
                    # Generate recommendations
                    with st.spinner("Generating recommendations..."):
                        recommendations = recommender.prioritize_folders()
                        st.session_state.recommendations = recommendations
                        
                        # Create visualizations
                        visualizations = recommender.visualize_recommendations()
                        st.session_state.visualizations = visualizations
                        
                        # Create summary table
                        summary_table = recommender.create_summary_table()
                        st.session_state.summary_table = summary_table
                        
                        st.session_state.analysis_complete = True
                else:
                    st.write("No users exceed the threshold. No recommendations needed.")
            
            # Show recommendations if analysis is complete
            if st.session_state.analysis_complete:
                st.subheader("Recommendations Summary")
                st.dataframe(st.session_state.summary_table)
                
                # Show visualizations for each user
                st.subheader("Visualizations")
                
                # Create tabs for each user
                if st.session_state.visualizations:
                    user_tabs = st.tabs(list(st.session_state.visualizations.keys()))
                    
                    for i, (user_email, user_viz) in enumerate(st.session_state.visualizations.items()):
                        with user_tabs[i]:
                            # Show recommendations for this user
                            user_recs = st.session_state.recommendations[user_email]
                            
                            st.write(f"**Original File Count:** {user_recs['total_file_count']:,}")
                            st.write(f"**Final File Count:** {user_recs['final_file_count']:,}")
                            st.write(f"**Files to Move:** {user_recs['total_recommended_moves']:,}")
                            st.write(f"**Folders to Split:** {len(user_recs['recommended_splits'])}")
                            st.write(f"**Service Accounts Needed:** {len(user_recs['service_accounts'])}")
                            
                            # Show visualizations
                            viz_tabs = st.tabs([
                                "Recommended Splits", 
                                "Before vs. After", 
                                "Service Account Distribution",
                                "Folder Distribution",
                                "Size Distribution",
                                "Level Distribution"
                            ])
                            
                            with viz_tabs[0]:
                                if 'recommendations' in user_viz:
                                    st.pyplot(user_viz['recommendations'])
                                    
                                    # Show table of recommended splits
                                    splits_df = pd.DataFrame([
                                        {
                                            'Folder Path': split['folder_path'],
                                            'Folder Name': split['folder_name'],
                                            'Level': split['level'],
                                            'Current Files': split['current_file_count'],
                                            'Files to Move': split['recommended_files_to_move'],
                                            'Partial Split': 'Yes' if split.get('is_partial_split', False) else 'No',
                                            'Assigned To': split.get('assigned_to', '')
                                        }
                                        for split in user_recs['recommended_splits']
                                    ])
                                    st.dataframe(splits_df)
                            
                            with viz_tabs[1]:
                                if 'before_after' in user_viz:
                                    st.pyplot(user_viz['before_after'])
                            
                            with viz_tabs[2]:
                                if 'service_account_distribution' in user_viz:
                                    st.pyplot(user_viz['service_account_distribution'])
                                    
                                    # Show table of service accounts
                                    accounts_df = pd.DataFrame([
                                        {
                                            'Account': account['account_name'],
                                            'Total Files': account['total_files'],
                                            'Folder Count': len(account['folders']),
                                            'Utilization': f"{account['total_files'] / file_threshold * 100:.1f}%"
                                        }
                                        for account in user_recs['service_accounts']
                                    ])
                                    st.dataframe(accounts_df)
                            
                            with viz_tabs[3]:
                                if 'folder_distribution' in user_viz:
                                    st.pyplot(user_viz['folder_distribution'])
                            
                            with viz_tabs[4]:
                                if 'size_distribution' in user_viz:
                                    st.pyplot(user_viz['size_distribution'])
                            
                            with viz_tabs[5]:
                                if 'level_distribution' in user_viz:
                                    st.pyplot(user_viz['level_distribution'])
                
                # Export options
                st.subheader("Export Recommendations")
                
                if st.button("Generate Export Files"):
                    with st.spinner("Generating export files..."):
                        exports = recommender.export_recommendations()
                        
                        # Download buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="Download JSON",
                                data=exports['json'],
                                file_name="recommendations.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            st.download_button(
                                label="Download CSV",
                                data=exports['csv'],
                                file_name="recommendations.csv",
                                mime="text/csv"
                            )
                        
                        with col3:
                            st.download_button(
                                label="Download Excel",
                                data=base64.b64decode(exports['excel']),
                                file_name="recommendations.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        st.download_button(
                            label="Download All (ZIP)",
                            data=base64.b64decode(exports['zip']),
                            file_name="folder_split_recommendations.zip",
                            mime="application/zip"
                        )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error in main app: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
