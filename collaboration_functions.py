import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import streamlit as st
from typing import Dict, List, Tuple, Any

def analyze_collaborations(collab_df: pd.DataFrame, folder_splits_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the impact of folder splits on collaborations.
    
    Args:
        collab_df: DataFrame containing collaboration data with required columns:
            - item_id: ID of the folder (used to match with Folder ID in folder data)
            - item_type: Type of item being collaborated on
            - collaborator_id: ID of the collaborator
            - collaborator_name: Name of the collaborator
            - collaborator_login: Login of the collaborator
            - collaborator_type: Type of collaborator
            - collaborator_permission: Permission level of the collaborator
            - collaboration_id: ID of the collaboration
        folder_splits_df: DataFrame containing folder split recommendations
    
    Returns:
        Dictionary containing collaboration analysis results:
        - maintained_collaborations: DataFrame of collaborations that will be maintained
        - lost_collaborations: DataFrame of collaborations that will be lost
        - remediation_actions: DataFrame of actions needed to maintain access
    """
    try:
        # Validate input data
        if collab_df.empty or folder_splits_df.empty:
            st.warning("No data available for collaboration analysis.")
            return {
                'maintained_collaborations': pd.DataFrame(),
                'lost_collaborations': pd.DataFrame(),
                'remediation_actions': pd.DataFrame()
            }
        
        # Ensure folder_splits_df has the required columns
        required_columns = ['Folder ID', 'Level', 'Service Account']
        missing_columns = [col for col in required_columns if col not in folder_splits_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in folder splits data: {', '.join(missing_columns)}")
            return {
                'maintained_collaborations': pd.DataFrame(),
                'lost_collaborations': pd.DataFrame(),
                'remediation_actions': pd.DataFrame()
            }
        
        # Create a mapping of folder IDs to their levels and service accounts
        folder_mapping = {}
        for _, row in folder_splits_df.iterrows():
            folder_id = row['Folder ID']
            folder_mapping[folder_id] = {
                'level': row['Level'],
                'service_account': row['Service Account']
            }
        
        # Initialize result DataFrames
        maintained_collaborations = []
        lost_collaborations = []
        remediation_actions = []
        
        # Process each collaboration
        for _, collab in collab_df.iterrows():
            item_id = collab['item_id']
            
            # Check if this item is in the folder splits
            if item_id in folder_mapping:
                folder_info = folder_mapping[item_id]
                level = folder_info['level']
                service_account = folder_info['service_account']
                
                # Apply collaboration rules
                if level == 1:  # Root-level folder
                    # For root-level folder splits: collaborations remain unchanged
                    maintained_collaborations.append({
                        'item_id': item_id,
                        'collaborator_id': collab['collaborator_id'],
                        'collaborator_name': collab['collaborator_name'],
                        'collaborator_login': collab['collaborator_login'],
                        'collaborator_type': collab['collaborator_type'],
                        'collaborator_permission': collab['collaborator_permission'],
                        'collaboration_id': collab['collaboration_id'],
                        'reason': "Root-level folder split - collaboration maintained"
                    })
                else:  # Subfolder
                    # For subfolder splits:
                    # - Collaborators added directly to the subfolder will see it in their root folders
                    # - Collaborators added at parent-folder level or above will lose access
                    
                    # Check if this is a direct collaboration on the subfolder
                    is_direct_collaboration = True  # Assume direct collaboration by default
                    
                    # In a real implementation, you would check if the collaboration was added
                    # directly to this folder or inherited from a parent folder
                    # For this example, we'll use a simple heuristic based on permission level
                    if collab['collaborator_permission'] in ['viewer', 'commenter']:
                        # Assume these lower-level permissions might be inherited
                        is_direct_collaboration = False
                    
                    if is_direct_collaboration:
                        maintained_collaborations.append({
                            'item_id': item_id,
                            'collaborator_id': collab['collaborator_id'],
                            'collaborator_name': collab['collaborator_name'],
                            'collaborator_login': collab['collaborator_login'],
                            'collaborator_type': collab['collaborator_type'],
                            'collaborator_permission': collab['collaborator_permission'],
                            'collaboration_id': collab['collaboration_id'],
                            'reason': "Direct collaboration on subfolder - will see in root folders"
                        })
                    else:
                        lost_collaborations.append({
                            'item_id': item_id,
                            'collaborator_id': collab['collaborator_id'],
                            'collaborator_name': collab['collaborator_name'],
                            'collaborator_login': collab['collaborator_login'],
                            'collaborator_type': collab['collaborator_type'],
                            'collaborator_permission': collab['collaborator_permission'],
                            'collaboration_id': collab['collaboration_id'],
                            'reason': "Parent-level collaboration - will lose access"
                        })
                        
                        # Add remediation action
                        remediation_actions.append({
                            'item_id': item_id,
                            'collaborator_id': collab['collaborator_id'],
                            'collaborator_name': collab['collaborator_name'],
                            'collaborator_login': collab['collaborator_login'],
                            'collaborator_type': collab['collaborator_type'],
                            'collaborator_permission': collab['collaborator_permission'],
                            'service_account': service_account,
                            'action': "Add collaborator directly to folder"
                        })
        
        # Convert lists to DataFrames
        maintained_df = pd.DataFrame(maintained_collaborations) if maintained_collaborations else pd.DataFrame()
        lost_df = pd.DataFrame(lost_collaborations) if lost_collaborations else pd.DataFrame()
        remediation_df = pd.DataFrame(remediation_actions) if remediation_actions else pd.DataFrame()
        
        return {
            'maintained_collaborations': maintained_df,
            'lost_collaborations': lost_df,
            'remediation_actions': remediation_df
        }
    
    except Exception as e:
        st.error(f"Error analyzing collaborations: {str(e)}")
        return {
            'maintained_collaborations': pd.DataFrame(),
            'lost_collaborations': pd.DataFrame(),
            'remediation_actions': pd.DataFrame()
        }

def display_collaboration_results(collaboration_analysis: Dict[str, pd.DataFrame]) -> None:
    """
    Display the results of the collaboration analysis.
    
    Args:
        collaboration_analysis: Dictionary containing collaboration analysis results
    """
    try:
        maintained_df = collaboration_analysis.get('maintained_collaborations', pd.DataFrame())
        lost_df = collaboration_analysis.get('lost_collaborations', pd.DataFrame())
        remediation_df = collaboration_analysis.get('remediation_actions', pd.DataFrame())
        
        # Display summary statistics
        st.subheader("Collaboration Impact Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Maintained Collaborations", len(maintained_df))
        with col2:
            st.metric("Lost Collaborations", len(lost_df))
        with col3:
            st.metric("Remediation Actions", len(remediation_df))
        
        # Display detailed results in tabs
        tabs = st.tabs(["Maintained Collaborations", "Lost Collaborations", "Remediation Actions"])
        
        with tabs[0]:
            if not maintained_df.empty:
                st.dataframe(maintained_df)
                
                # Download CSV button
                csv = maintained_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="maintained_collaborations.csv">Download Maintained Collaborations CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No maintained collaborations to display.")
        
        with tabs[1]:
            if not lost_df.empty:
                st.dataframe(lost_df)
                
                # Download CSV button
                csv = lost_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="lost_collaborations.csv">Download Lost Collaborations CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Create visualization of lost collaborations
                if len(lost_df) > 0:
                    st.subheader("Lost Collaborations by Permission Level")
                    
                    # Group by permission level
                    permission_counts = lost_df['collaborator_permission'].value_counts()
                    
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(permission_counts, labels=permission_counts.index, autopct='%1.1f%%', 
                           startangle=90, colors=plt.cm.tab10.colors[:len(permission_counts)])
                    ax.axis('equal')
                    st.pyplot(fig)
            else:
                st.info("No lost collaborations to display.")
        
        with tabs[2]:
            if not remediation_df.empty:
                st.dataframe(remediation_df)
                
                # Download CSV button
                csv = remediation_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="remediation_actions.csv">Download Remediation Actions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Create visualization of remediation actions
                if len(remediation_df) > 0:
                    st.subheader("Remediation Actions by Service Account")
                    
                    # Group by service account
                    service_account_counts = remediation_df['service_account'].value_counts()
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(service_account_counts.index, service_account_counts.values, color='skyblue')
                    
                    # Add data labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    ax.set_xlabel('Service Account')
                    ax.set_ylabel('Number of Actions')
                    ax.set_title('Remediation Actions by Service Account')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No remediation actions to display.")
    
    except Exception as e:
        st.error(f"Error displaying collaboration results: {str(e)}")
