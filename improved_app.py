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
# New session state variables for collaboration data
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
    """
    
    def __init__(self, df: pd.DataFrame, file_threshold: int = 500000, collaboration_df: pd.DataFrame = None):
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
                - item_id: ID to match with Folder ID in the original upload
                - item_type: Type of item
                - collaborator_id: ID of the collaborator
                - collaborator_name: Name of the collaborator
                - collaborator_login: Login of the collaborator
                - collaborator_type: Type of collaborator
                - collaborator_permission: Permission level of the collaborator
                - collaboration_id: ID of the collaboration
        """
        self.df = df
        self.file_threshold = file_threshold
        self.collaboration_df = collaboration_df
        # Set service account threshold to 490,000 (hard requirement)
        self.service_account_threshold = 490000
        self.users_exceeding = None
        self.recommendations = {}
        self.collaboration_analysis = {}
        
        # Validate input data
        self._validate_input_data()
        
        # Process collaboration data if provided
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
        required_columns = [
            'item_id', 'item_type', 'collaborator_id', 'collaborator_name', 
            'collaborator_login', 'collaborator_type', 'collaborator_permission', 'collaboration_id'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in self.collaboration_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns in collaboration data: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check data types and convert if necessary
        try:
            # Ensure item_id is numeric for matching with Folder ID
            self.collaboration_df['item_id'] = pd.to_numeric(self.collaboration_df['item_id'])
            
            # Ensure string columns are strings
            string_columns = ['item_type', 'collaborator_name', 'collaborator_login', 
                             'collaborator_type', 'collaborator_permission']
            for col in string_columns:
                self.collaboration_df[col] = self.collaboration_df[col].astype(str)
        except Exception as e:
            error_msg = f"Error converting collaboration data types: {str(e)}"
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
            
            # Create a mapping of folder paths to their folder objects
            folder_map = {folder['folder_path']: folder for folder in folders_to_split}
            
            # Create a mapping to track which service account each folder is assigned to
            folder_to_account = {}
            
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
                    
                    # If this folder is related to an already assigned folder, try to assign to the same account
                    if assigned_to_account is not None:
                        # Find the service account
                        for account in service_accounts:
                            if account['account_name'] == assigned_to_account:
                                # Check if adding this folder would exceed the threshold
                                new_total = account['total_files'] + folder['recommended_files_to_move']
                                if new_total <= self.service_account_threshold:
                                    # Add to this account
                                    account['folders'].append(folder)
                                    account['total_files'] = new_total
                                    folder_to_account[folder_path] = account['account_name']
                                    folder['assigned_to'] = account['account_name']
                                    logger.info(f"Added folder {folder_path} to existing account {account['account_name']}, new total: {new_total:,}")
                                    break
                                else:
                                    # This account would exceed threshold, need to create a new one
                                    logger.info(f"Cannot add folder {folder_path} to account {account['account_name']} (would exceed threshold)")
                                    assigned_to_account = None
                    
                    # If not assigned to an existing account, try to add to current account or create a new one
                    if assigned_to_account is None:
                        # Check if adding to current account would exceed threshold
                        new_total = current_account['total_files'] + folder['recommended_files_to_move']
                        if new_total <= self.service_account_threshold:
                            # Add to current account
                            current_account['folders'].append(folder)
                            current_account['total_files'] = new_total
                            folder_to_account[folder_path] = current_account['account_name']
                            folder['assigned_to'] = current_account['account_name']
                            logger.info(f"Added folder {folder_path} to current account {current_account['account_name']}, new total: {new_total:,}")
                        else:
                            # Current account would exceed threshold, create a new one
                            if current_account['folders']:  # Only add if it has folders
                                service_accounts.append(current_account)
                            
                            # Create a new account
                            account_num = len(service_accounts) + 1
                            current_account = {
                                'account_name': f'service_account_{account_num}',
                                'folders': [folder],
                                'total_files': folder['recommended_files_to_move']
                            }
                            folder_to_account[folder_path] = current_account['account_name']
                            folder['assigned_to'] = current_account['account_name']
                            logger.info(f"Created new account {current_account['account_name']} for folder {folder_path}, initial total: {folder['recommended_files_to_move']:,}")
            
            # Add the last account if it has folders
            if current_account['folders']:
                service_accounts.append(current_account)
            
            logger.info(f"Created {len(service_accounts)} service accounts for user {user_email}")
            return service_accounts
            
        except Exception as e:
            error_msg = f"Error assigning to service accounts: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def prioritize_folders(self) -> Dict[str, Dict[str, Any]]:
        """
        Prioritize folders for splitting based on file count and folder structure.
        
        This method identifies which folders should be split for each user exceeding the threshold.
        It prioritizes larger folders first to maximize the reduction in file count while minimizing
        the number of splits needed.
        
        Returns:
            Dictionary of recommendations for each user
        """
        logger.info("Prioritizing folders for splitting...")
        st.write("Prioritizing folders for splitting...")
        
        try:
            # Ensure we have user stats
            if self.users_exceeding is None:
                self.calculate_user_stats()
            
            # Ensure we have direct file counts
            if 'direct_file_count' not in self.df.columns:
                self.identify_nested_folders()
            
            # For each user exceeding the threshold
            for _, user_row in self.users_exceeding.iterrows():
                user_email = user_row['Owner']
                total_file_count = user_row['total_file_count']
                
                logger.info(f"Processing user: {user_email} with {total_file_count:,} files")
                st.write(f"Processing user: {user_email} with {total_file_count:,} files")
                
                # Initialize recommendations for this user
                recommendations = {
                    'user_email': user_email,
                    'total_file_count': int(total_file_count),
                    'file_threshold': self.file_threshold,
                    'excess_files': int(total_file_count - self.file_threshold),
                    'recommended_splits': [],
                    'unsplittable_folders': [],
                    'total_recommended_moves': 0,
                    'final_file_count': int(total_file_count),
                    'remaining_excess_files': int(total_file_count - self.file_threshold)
                }
                
                # Get all folders owned by this user
                user_folders = self.df[self.df['Owner'] == user_email].copy()
                
                # Calculate excess files (how many files need to be moved)
                excess_files = total_file_count - self.file_threshold
                
                # Track remaining files after splits
                remaining_files = total_file_count
                
                # Track which folders have been selected for splitting
                selected_folder_paths = []
                
                # Process level 1 folders first (direct children of root)
                level1_folders = user_folders[user_folders['Level'] == 1].sort_values('File Count', ascending=False)
                
                # Find folders that exceed the threshold and cannot be split
                for _, folder in level1_folders.iterrows():
                    folder_file_count = folder['File Count']
                    
                    # Skip small folders
                    if folder_file_count < 10000:  # Skip folders with fewer than 10,000 files
                        continue
                    
                    # Check if this folder has any subfolders
                    folder_path = folder['Path']
                    has_subfolders = any(
                        path.startswith(folder_path) and path != folder_path 
                        for path in user_folders['Path']
                    )
                    
                    # If this folder exceeds threshold and has no subfolders, it cannot be split
                    if folder_file_count > self.file_threshold and not has_subfolders:
                        recommendations['unsplittable_folders'].append({
                            'folder_path': folder_path,
                            'folder_name': folder['Folder Name'],
                            'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                            'level': int(folder['Level']),
                            'file_count': int(folder_file_count),
                            'reason': "Exceeds threshold and cannot be split (no subfolders)"
                        })
                
                # Find candidate folders for splitting
                candidates = []
                
                # Consider all folders (any level) as candidates
                for _, folder in user_folders.sort_values('File Count', ascending=False).iterrows():
                    folder_path = folder['Path']
                    folder_file_count = folder['File Count']
                    
                    # Skip small folders
                    if folder_file_count < 10000:  # Skip folders with fewer than 10,000 files
                        continue
                    
                    # Skip if this folder is a subfolder of any already selected folder
                    if self._is_subfolder_of_any(folder_path, selected_folder_paths):
                        continue
                    
                    # Skip if this folder is a parent of any already selected folder
                    if self._is_parent_of_any(folder_path, selected_folder_paths):
                        continue
                    
                    # Check if this folder has any subfolders
                    has_subfolders = any(
                        path.startswith(folder_path) and path != folder_path 
                        for path in user_folders['Path']
                    )
                    
                    # If this folder exceeds threshold and has no subfolders, it cannot be split
                    if folder_file_count > self.file_threshold and not has_subfolders:
                        # Skip if already added to unsplittable_folders
                        if any(f['folder_path'] == folder_path for f in recommendations['unsplittable_folders']):
                            continue
                            
                        recommendations['unsplittable_folders'].append({
                            'folder_path': folder_path,
                            'folder_name': folder['Folder Name'],
                            'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                            'level': int(folder['Level']),
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
                        'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                        'level': int(folder['Level']),
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
                        'current_file_count': folder['current_file_count'],
                        'direct_file_count': folder['direct_file_count'],
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
                
                # Set the correct values in the recommendations
                recommendations['total_recommended_moves'] = int(total_files_moved)
                recommendations['final_file_count'] = int(final_file_count)
                recommendations['remaining_excess_files'] = int(final_file_count - self.file_threshold) if final_file_count > self.file_threshold else 0
                
                # Save recommendations for this user
                self.recommendations[user_email] = recommendations
                
                logger.info(f"Final recommendations for {user_email}:")
                logger.info(f"Total files moved to service accounts: {total_files_moved:,}")
                logger.info(f"Final file count after all splits: {final_file_count:,}")
                logger.info(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
                logger.info(f"Number of service accounts needed: {len(service_accounts)}")
                
                st.write(f"Final recommendations for {user_email}:")
                st.write(f"Total files moved to service accounts: {total_files_moved:,}")
                st.write(f"Final file count after all splits: {final_file_count:,}")
                st.write(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
                st.write(f"Number of service accounts needed: {len(service_accounts)}")
            
            return self.recommendations
            
        except Exception as e:
            error_msg = f"Error prioritizing folders: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def analyze_collaborations(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze collaboration data and determine which collaborators will be affected by folder splits.
        
        This method implements the collaboration logic:
        - For root-level folder splits: collaborations remain unchanged
        - For subfolder splits:
          - Collaborators added directly to the subfolder will see it in their root folders
          - Collaborators added at parent-folder level or above will lose access
        
        Returns:
            Dictionary containing collaboration analysis results
        """
        logger.info("Analyzing collaboration data...")
        st.write("Analyzing collaboration data...")
        
        try:
            # Check if we have collaboration data
            if self.collaboration_df is None:
                logger.info("No collaboration data provided, skipping collaboration analysis")
                return {}
            
            # Check if we have recommendations
            if not self.recommendations:
                logger.info("No recommendations available, skipping collaboration analysis")
                return {}
            
            # Create a mapping of folder IDs to folder paths
            folder_id_to_path = {}
            folder_id_to_level = {}
            for _, row in self.df.iterrows():
                if not pd.isna(row['Folder ID']):
                    folder_id = int(row['Folder ID'])
                    folder_id_to_path[folder_id] = row['Path']
                    folder_id_to_level[folder_id] = int(row['Level'])
            
            # Create a mapping of item_id to collaborators
            item_to_collaborators = {}
            for _, row in self.collaboration_df.iterrows():
                item_id = int(row['item_id'])
                if item_id not in item_to_collaborators:
                    item_to_collaborators[item_id] = []
                
                item_to_collaborators[item_id].append({
                    'collaborator_id': row['collaborator_id'],
                    'collaborator_name': row['collaborator_name'],
                    'collaborator_login': row['collaborator_login'],
                    'collaborator_type': row['collaborator_type'],
                    'collaborator_permission': row['collaborator_permission'],
                    'collaboration_id': row['collaboration_id']
                })
            
            # For each user with recommendations
            for user_email, user_recs in self.recommendations.items():
                # Initialize collaboration analysis for this user
                self.collaboration_analysis[user_email] = {
                    'user_email': user_email,
                    'collaborators_to_add': [],
                    'collaborators_losing_access': []
                }
                
                # Process each recommended split
                for folder in user_recs['recommended_splits']:
                    folder_id = folder['folder_id']
                    folder_level = folder['level']
                    
                    # Skip if folder ID is not available
                    if folder_id is None:
                        continue
                    
                    # Get collaborators for this folder
                    folder_collaborators = item_to_collaborators.get(folder_id, [])
                    
                    # For root-level folder splits (level 1), collaborations remain unchanged
                    if folder_level == 1:
                        logger.info(f"Root-level folder split for {folder['folder_path']}, collaborations remain unchanged")
                        continue
                    
                    # For subfolder splits (level > 1)
                    # Find parent folder IDs
                    parent_folder_ids = []
                    folder_path = folder['folder_path']
                    path_parts = folder_path.strip('/').split('/')
                    
                    # Build potential parent paths
                    for i in range(1, len(path_parts)):
                        parent_path = '/' + '/'.join(path_parts[:i]) + '/'
                        # Find folder ID for this parent path
                        for fid, path in folder_id_to_path.items():
                            if path == parent_path:
                                parent_folder_ids.append(fid)
                    
                    # Get collaborators for parent folders
                    parent_collaborators = []
                    for parent_id in parent_folder_ids:
                        parent_collaborators.extend(item_to_collaborators.get(parent_id, []))
                    
                    # Collaborators added directly to the subfolder will see it in their root folders
                    # Add them to the list of collaborators to add
                    for collaborator in folder_collaborators:
                        self.collaboration_analysis[user_email]['collaborators_to_add'].append({
                            'folder_id': folder_id,
                            'folder_name': folder['folder_name'],
                            'folder_path': folder['folder_path'],
                            'collaborator_id': collaborator['collaborator_id'],
                            'collaborator_name': collaborator['collaborator_name'],
                            'collaborator_login': collaborator['collaborator_login'],
                            'collaborator_type': collaborator['collaborator_type'],
                            'collaborator_permission': collaborator['collaborator_permission'],
                            'collaboration_id': collaborator['collaboration_id'],
                            'service_account': folder.get('assigned_to', 'Unknown')
                        })
                    
                    # Collaborators added at parent-folder level or above will lose access
                    # Add them to the list of collaborators losing access
                    for collaborator in parent_collaborators:
                        # Check if this collaborator is not already directly added to the subfolder
                        if not any(c['collaborator_id'] == collaborator['collaborator_id'] for c in folder_collaborators):
                            self.collaboration_analysis[user_email]['collaborators_losing_access'].append({
                                'folder_id': folder_id,
                                'folder_name': folder['folder_name'],
                                'folder_path': folder['folder_path'],
                                'collaborator_id': collaborator['collaborator_id'],
                                'collaborator_name': collaborator['collaborator_name'],
                                'collaborator_login': collaborator['collaborator_login'],
                                'collaborator_type': collaborator['collaborator_type'],
                                'collaborator_permission': collaborator['collaborator_permission'],
                                'collaboration_id': collaborator['collaboration_id']
                            })
                
                # Log summary
                logger.info(f"Collaboration analysis for {user_email}:")
                logger.info(f"Collaborators to add: {len(self.collaboration_analysis[user_email]['collaborators_to_add'])}")
                logger.info(f"Collaborators losing access: {len(self.collaboration_analysis[user_email]['collaborators_losing_access'])}")
            
            return self.collaboration_analysis
            
        except Exception as e:
            error_msg = f"Error analyzing collaborations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def get_collaboration_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate tables for collaborators to add and collaborators losing access.
        
        Returns:
            Tuple containing:
            - DataFrame of collaborators to add
            - DataFrame of collaborators losing access
        """
        try:
            # Check if we have collaboration analysis
            if not self.collaboration_analysis:
                return pd.DataFrame(), pd.DataFrame()
            
            # Prepare data for collaborators to add
            collaborators_to_add_data = []
            for user_email, analysis in self.collaboration_analysis.items():
                for collaborator in analysis['collaborators_to_add']:
                    collaborators_to_add_data.append({
                        'User': user_email,
                        'Folder ID': collaborator['folder_id'],
                        'Folder Name': collaborator['folder_name'],
                        'Folder Path': collaborator['folder_path'],
                        'Collaborator ID': collaborator['collaborator_id'],
                        'Collaborator Name': collaborator['collaborator_name'],
                        'Collaborator Email': collaborator['collaborator_login'],
                        'Collaborator Type': collaborator['collaborator_type'],
                        'Permission': collaborator['collaborator_permission'],
                        'Service Account': collaborator['service_account']
                    })
            
            # Prepare data for collaborators losing access
            collaborators_losing_access_data = []
            for user_email, analysis in self.collaboration_analysis.items():
                for collaborator in analysis['collaborators_losing_access']:
                    collaborators_losing_access_data.append({
                        'User': user_email,
                        'Folder ID': collaborator['folder_id'],
                        'Folder Name': collaborator['folder_name'],
                        'Folder Path': collaborator['folder_path'],
                        'Collaborator ID': collaborator['collaborator_id'],
                        'Collaborator Name': collaborator['collaborator_name'],
                        'Collaborator Email': collaborator['collaborator_login'],
                        'Collaborator Type': collaborator['collaborator_type'],
                        'Permission': collaborator['collaborator_permission']
                    })
            
            # Create DataFrames
            collaborators_to_add_df = pd.DataFrame(collaborators_to_add_data)
            collaborators_losing_access_df = pd.DataFrame(collaborators_losing_access_data)
            
            return collaborators_to_add_df, collaborators_losing_access_df
            
        except Exception as e:
            error_msg = f"Error generating collaboration tables: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return pd.DataFrame(), pd.DataFrame()
    
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
                
                # 6. Plot folder size distribution
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create size bins
                size_bins = [0, 10000, 50000, 100000, 250000, 500000, 1000000, float('inf')]
                size_labels = ['<10K', '10K-50K', '50K-100K', '100K-250K', '250K-500K', '500K-1M', '>1M']
                
                # Count folders in each bin
                size_counts = [0] * len(size_labels)
                for _, folder in splits_df.iterrows():
                    for i, upper in enumerate(size_bins[1:]):
                        if folder['current_file_count'] < upper:
                            size_counts[i] += 1
                            break
                
                # Create bar chart
                ax.bar(size_labels, size_counts, color='skyblue')
                
                # Add data labels
                for i, count in enumerate(size_counts):
                    if count > 0:
                        ax.text(i, count + 0.1, str(count), ha='center')
                
                ax.set_xlabel('Folder Size (Files)')
                ax.set_ylabel('Number of Folders')
                ax.set_title(f'Folder Size Distribution for {user_email}')
                plt.tight_layout()
                user_visualizations['size_distribution'] = fig
                
                # 7. Plot folder level distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Count folders at each level
                level_counts = splits_df['level'].value_counts().sort_index()
                
                # Create bar chart
                ax.bar(level_counts.index, level_counts.values, color='lightgreen')
                
                # Add data labels
                for i, count in enumerate(level_counts.values):
                    if count > 0:
                        ax.text(level_counts.index[i], count + 0.1, str(count), ha='center')
                
                ax.set_xlabel('Folder Level')
                ax.set_ylabel('Number of Folders')
                ax.set_title(f'Folder Level Distribution for {user_email}')
                ax.set_xticks(level_counts.index)
                plt.tight_layout()
                user_visualizations['level_distribution'] = fig
                
                # 8. NEW: Plot collaboration impact if collaboration data is available
                if self.collaboration_analysis and user_email in self.collaboration_analysis:
                    collab_analysis = self.collaboration_analysis[user_email]
                    
                    # Count collaborators to add and losing access
                    collab_to_add = len(collab_analysis['collaborators_to_add'])
                    collab_losing = len(collab_analysis['collaborators_losing_access'])
                    
                    if collab_to_add > 0 or collab_losing > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        labels = ['Collaborators to Add', 'Collaborators Losing Access']
                        values = [collab_to_add, collab_losing]
                        colors = ['#66b3ff', '#ff9999']
                        
                        ax.bar(labels, values, color=colors)
                        
                        # Add data labels
                        for i, v in enumerate(values):
                            ax.text(i, v + v*0.02, str(v), ha='center')
                        
                        ax.set_ylabel('Number of Collaborators')
                        ax.set_title(f'Collaboration Impact for {user_email}')
                        plt.tight_layout()
                        user_visualizations['collaboration_impact'] = fig
                
                # Store visualizations for this user
                visualizations[user_email] = user_visualizations
            
            return visualizations
            
        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table of recommendations.
        
        Returns:
            DataFrame containing summary information for all users and service accounts
        """
        try:
            # Prepare data for summary table
            summary_data = []
            
            # For each user with recommendations
            for user_email, user_recs in self.recommendations.items():
                # Add row for original user
                summary_data.append({
                    'User': user_email,
                    'Before Split': user_recs['total_file_count'],
                    'After All Splits': user_recs['final_file_count'],
                    'Files Moved': user_recs['total_recommended_moves'],
                    'Remaining Excess': user_recs['remaining_excess_files'],
                    'Service Accounts': len(user_recs.get('service_accounts', [])),
                    'Folders Split': len(user_recs['recommended_splits']),
                    'Unsplittable Folders': len(user_recs.get('unsplittable_folders', []))
                })
                
                # Add rows for service accounts
                if 'service_accounts' in user_recs:
                    for account in user_recs['service_accounts']:
                        summary_data.append({
                            'User': account['account_name'],
                            'Before Split': 0,  # Service accounts start with 0 files
                            'After All Splits': account['total_files'],
                            'Files Moved': account['total_files'],
                            'Remaining Excess': max(0, account['total_files'] - self.file_threshold),
                            'Service Accounts': 0,  # Service accounts don't have their own service accounts
                            'Folders Split': 0,  # Service accounts don't split folders
                            'Unsplittable Folders': 0  # Service accounts don't have unsplittable folders
                        })
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Sort by user type (original users first, then service accounts)
            summary_df['is_service_account'] = summary_df['User'].str.contains('service_account')
            summary_df = summary_df.sort_values(['is_service_account', 'User'])
            summary_df = summary_df.drop(columns=['is_service_account'])
            
            return summary_df
            
        except Exception as e:
            error_msg = f"Error generating summary table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def get_folder_splits_table(self) -> pd.DataFrame:
        """
        Generate a detailed table of folder splits.
        
        Returns:
            DataFrame containing detailed information for all folder splits
        """
        try:
            # Prepare data for folder splits table
            splits_data = []
            
            # For each user with recommendations
            for user_email, user_recs in self.recommendations.items():
                # For each recommended split
                for folder in user_recs['recommended_splits']:
                    splits_data.append({
                        'User': user_email,
                        'Folder ID': folder['folder_id'],
                        'Folder Name': folder['folder_name'],
                        'Folder Path': folder['folder_path'],
                        'Level': folder['level'],
                        'Current File Count': folder['current_file_count'],
                        'Direct File Count': folder['direct_file_count'],
                        'Files to Move': folder['recommended_files_to_move'],
                        'New Total After Split': folder['new_total_after_split'],
                        'Service Account': folder.get('assigned_to', 'Unknown'),
                        'Split Type': 'Partial Split' if folder.get('is_partial_split', False) else 'Complete Split'
                    })
            
            # Create DataFrame
            splits_df = pd.DataFrame(splits_data)
            
            # Sort by user and file count (descending)
            if not splits_df.empty:
                splits_df = splits_df.sort_values(['User', 'Current File Count'], ascending=[True, False])
            
            return splits_df
            
        except Exception as e:
            error_msg = f"Error generating folder splits table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise

def main():
    """Main function to run the Streamlit app."""
    # Set up the app header
    st.title("Folder Splitting Recommendation Tool")
    st.markdown("""
    This tool analyzes folder ownership data and provides recommendations for splitting content
    to bring users below the file count threshold.
    """)
    
    # Create tabs for different sections
    tabs = st.tabs(["Upload & Analysis", "Results", "About"])
    
    with tabs[0]:
        st.header("Upload Data")
        
        # File uploader for folder data
        st.subheader("1. Upload Folder Data")
        uploaded_file = st.file_uploader(
            "Upload a CSV file containing folder data",
            type=["csv"],
            help="The CSV file should contain columns: Path, Folder Name, Folder ID, Owner, Size (MB), File Count, Level"
        )
        
        # NEW: File uploader for collaboration data
        st.subheader("2. Upload Collaboration Data (Optional)")
        collaboration_file = st.file_uploader(
            "Upload a CSV file containing collaboration data",
            type=["csv"],
            help="The CSV file should contain columns: item_id, item_type, collaborator_id, collaborator_name, collaborator_login, collaborator_type, collaborator_permission, collaboration_id"
        )
        
        # Threshold input
        st.subheader("3. Set File Count Threshold")
        threshold = st.number_input(
            "Maximum number of files a user should have",
            min_value=100000,
            max_value=10000000,
            value=500000,
            step=50000,
            help="Users with more than this number of files will receive recommendations for splitting"
        )
        
        # Store threshold in session state
        st.session_state.threshold = threshold
        
        # Generate recommendations button
        st.subheader("4. Generate Recommendations")
        generate_button = st.button("Generate Recommendations")
        
        # Process data when button is clicked
        if generate_button and uploaded_file is not None:
            try:
                # Display progress bar
                progress_bar = st.progress(0)
                st.write("Reading data...")
                
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                progress_bar.progress(0.1)
                
                # Read collaboration data if provided
                collaboration_df = None
                if collaboration_file is not None:
                    st.write("Reading collaboration data...")
                    collaboration_df = pd.read_csv(collaboration_file)
                    progress_bar.progress(0.15)
                
                # Process data
                users_exceeding, recommendations, visualizations, summary_table = process_data(df, threshold, collaboration_df, progress_bar)
                
                # Store results in session state
                st.session_state.users_exceeding = users_exceeding
                st.session_state.recommendations = recommendations
                st.session_state.visualizations = visualizations
                st.session_state.summary_table = summary_table
                
                # Mark analysis as complete
                st.session_state.analysis_complete = True
                
                # Display results
                display_results()
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.exception(e)
        else:
            st.info("Please upload a CSV file to begin analysis.")
            
    with tabs[1]:
        st.header("Results")
        
        # Check if analysis is complete
        if st.session_state.analysis_complete:
            display_results()
        else:
            st.info("No analysis results to display. Please generate recommendations first.")
    
    with tabs[2]:
        st.header("About")
        st.markdown("""
        ### Folder Splitting Recommendation Tool
        
        This tool helps identify users who exceed the file count threshold and recommends which folders to split
        to bring them below the threshold. It assigns excess folders to service accounts, ensuring each service
        account stays under the threshold.
        
        #### Features:
        - Analyzes folder ownership data to identify users exceeding the threshold
        - Recommends which folders to split based on file count and folder structure
        - Assigns folders to service accounts, keeping related folders together
        - Provides visualizations of the recommendations
        - Analyzes collaboration data to determine which collaborators will be affected by folder splits
        - Generates downloadable reports for implementation
        
        #### How to use:
        1. Upload a CSV file containing folder data
        2. (Optional) Upload a CSV file containing collaboration data
        3. Set the file count threshold
        4. Click "Generate Recommendations"
        5. View the results and download the reports
        
        #### Collaboration Logic:
        - For root-level folder splits: collaborations remain unchanged
        - For subfolder splits:
          - Collaborators added directly to the subfolder will see it in their root folders
          - Collaborators added at parent-folder level or above will lose access
        
        #### Required Columns for Folder Data:
        - Path: Folder path separated by "/"
        - Folder Name: Name of the folder
        - Folder ID: Integer ID of the folder
        - Owner: Email of the user that owns the folder
        - Size (MB): Size of the folder in MB
        - File Count: Number of active files within the folder and all subfolders
        - Level: Folder level in the folder tree hierarchy (1 is root level)
        
        #### Required Columns for Collaboration Data:
        - item_id: ID to match with Folder ID in the original upload
        - item_type: Type of item
        - collaborator_id: ID of the collaborator
        - collaborator_name: Name of the collaborator
        - collaborator_login: Login of the collaborator
        - collaborator_type: Type of collaborator
        - collaborator_permission: Permission level of the collaborator
        - collaboration_id: ID of the collaboration
        """)

def process_data(df: pd.DataFrame, threshold: int, collaboration_df: pd.DataFrame = None, progress_bar=None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, plt.Figure]], pd.DataFrame]:
    """
    Process the data and generate recommendations.
    
    Args:
        df: DataFrame containing folder data
        threshold: Maximum number of files a user should have
        collaboration_df: Optional DataFrame containing collaboration data
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
        recommender = FolderSplitRecommender(df, file_threshold=threshold, collaboration_df=collaboration_df)
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
        
        # Identify nested folders
        recommender.identify_nested_folders()
        if progress_bar:
            progress_bar.progress(0.4)
        
        # Prioritize folders
        recommendations = recommender.prioritize_folders()
        if progress_bar:
            progress_bar.progress(0.6)
        
        # Analyze collaborations if collaboration data is provided
        if collaboration_df is not None:
            collaboration_analysis = recommender.analyze_collaborations()
            st.session_state.collaboration_analysis = collaboration_analysis
            if progress_bar:
                progress_bar.progress(0.7)
        
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
        
        # Get collaboration tables if collaboration data is provided
        if collaboration_df is not None:
            collaborators_to_add_df, collaborators_losing_access_df = recommender.get_collaboration_tables()
            st.session_state.collaborators_to_add_df = collaborators_to_add_df
            st.session_state.collaborators_losing_access_df = collaborators_losing_access_df
        
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
    href = f'<a href="data:file/csv;base64,{b64}" download="summary.csv">Download Summary CSV</a>'
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
    
    # NEW: Display collaboration analysis if available
    if hasattr(st.session_state, 'collaborators_to_add_df') and hasattr(st.session_state, 'collaborators_losing_access_df'):
        collaborators_to_add_df = st.session_state.collaborators_to_add_df
        collaborators_losing_access_df = st.session_state.collaborators_losing_access_df
        
        if not collaborators_to_add_df.empty or not collaborators_losing_access_df.empty:
            st.subheader("Collaboration Analysis")
            
            # Create tabs for collaboration analysis
            collab_tabs = st.tabs(["Collaborators to Add", "Collaborators Losing Access"])
            
            with collab_tabs[0]:
                if not collaborators_to_add_df.empty:
                    st.write("These collaborators need to be added to the new folders:")
                    st.dataframe(collaborators_to_add_df)
                    
                    # Download CSV button for collaborators to add
                    csv_to_add = collaborators_to_add_df.to_csv(index=False)
                    b64_to_add = base64.b64encode(csv_to_add.encode()).decode()
                    href_to_add = f'<a href="data:file/csv;base64,{b64_to_add}" download="collaborators_to_add.csv">Download Collaborators to Add CSV</a>'
                    st.markdown(href_to_add, unsafe_allow_html=True)
                else:
                    st.info("No collaborators need to be added.")
            
            with collab_tabs[1]:
                if not collaborators_losing_access_df.empty:
                    st.write("These collaborators will lose access to the split folders:")
                    st.dataframe(collaborators_losing_access_df)
                    
                    # Download CSV button for collaborators losing access
                    csv_losing = collaborators_losing_access_df.to_csv(index=False)
                    b64_losing = base64.b64encode(csv_losing.encode()).decode()
                    href_losing = f'<a href="data:file/csv;base64,{b64_losing}" download="collaborators_losing_access.csv">Download Collaborators Losing Access CSV</a>'
                    st.markdown(href_losing, unsafe_allow_html=True)
                else:
                    st.info("No collaborators will lose access.")
    
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
                    'Percent of Threshold': f"{(account['total_files'] / st.session_state.threshold) * 100:.1f}%"
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
                        st.pyplot(user_viz['service_account_distribution'])
                    
                    if 'folder_distribution' in user_viz:
                        st.pyplot(user_viz['folder_distribution'])
                
                with viz_tabs[2]:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'size_distribution' in user_viz:
                            st.pyplot(user_viz['size_distribution'])
                    with col2:
                        if 'level_distribution' in user_viz:
                            st.pyplot(user_viz['level_distribution'])
                    
                    # NEW: Display collaboration impact visualization if available
                    if 'collaboration_impact' in user_viz:
                        st.pyplot(user_viz['collaboration_impact'])
            
            # NEW: Display collaboration analysis for this user if available
            if hasattr(st.session_state, 'collaboration_analysis'):
                collaboration_analysis = st.session_state.collaboration_analysis
                if user_email in collaboration_analysis:
                    user_collab = collaboration_analysis[user_email]
                    
                    st.subheader("Collaboration Analysis")
                    
                    # Create tabs for collaboration analysis
                    user_collab_tabs = st.tabs(["Collaborators to Add", "Collaborators Losing Access"])
                    
                    with user_collab_tabs[0]:
                        if user_collab['collaborators_to_add']:
                            # Create a DataFrame for collaborators to add
                            collab_add_data = []
                            for collab in user_collab['collaborators_to_add']:
                                collab_add_data.append({
                                    'Folder Name': collab['folder_name'],
                                    'Folder Path': collab['folder_path'],
                                    'Collaborator Name': collab['collaborator_name'],
                                    'Collaborator Email': collab['collaborator_login'],
                                    'Permission': collab['collaborator_permission'],
                                    'Service Account': collab['service_account']
                                })
                            
                            collab_add_df = pd.DataFrame(collab_add_data)
                            st.write(f"Collaborators to add ({len(collab_add_df)} total):")
                            st.dataframe(collab_add_df, use_container_width=True)
                        else:
                            st.info("No collaborators need to be added for this user.")
                    
                    with user_collab_tabs[1]:
                        if user_collab['collaborators_losing_access']:
                            # Create a DataFrame for collaborators losing access
                            collab_lose_data = []
                            for collab in user_collab['collaborators_losing_access']:
                                collab_lose_data.append({
                                    'Folder Name': collab['folder_name'],
                                    'Folder Path': collab['folder_path'],
                                    'Collaborator Name': collab['collaborator_name'],
                                    'Collaborator Email': collab['collaborator_login'],
                                    'Permission': collab['collaborator_permission']
                                })
                            
                            collab_lose_df = pd.DataFrame(collab_lose_data)
                            st.write(f"Collaborators losing access ({len(collab_lose_df)} total):")
                            st.dataframe(collab_lose_df, use_container_width=True)
                        else:
                            st.info("No collaborators will lose access for this user.")

if __name__ == "__main__":
    main()
