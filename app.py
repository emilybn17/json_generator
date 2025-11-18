import streamlit as st
import pandas as pd
import json
import numpy as np
from io import StringIO
import zipfile
from io import BytesIO
import re

st.set_page_config(page_title="Persona JSON Generator", page_icon="📋")

st.title("Persona JSON Generator")
st.write("Upload your artifact CSV files to generate a consolidated persona JSON")

# Helper functions
def json_serialize(obj):
    """Helper function to handle pandas/numpy data types for JSON"""
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def clean_dataframe(df):
    """Clean DataFrame for JSON compatibility and remove week column"""
    if 'week' in df.columns:
        df = df.drop('week', axis=1)
    df_clean = df.replace({np.nan: None})
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'bool':
            df_clean[col] = df_clean[col].map({True: True, False: False, None: None})
    
    return df_clean

def format_phone_number(phone):
    """Convert phone number to string format"""
    if pd.isna(phone) or phone is None:
        return None
    phone_str = str(phone)
    if phone_str.endswith('.0'):
        phone_str = phone_str[:-2]
    return phone_str

def detect_artifact_type(filename):
    """
    Detect artifact type from filename.
    Handles formats like: "Jordan Reid (template copy) - Contacts.csv"
    Returns lowercase key name for JSON
    """
    filename_lower = filename.lower()
    
    # Define mapping of keywords to JSON keys
    type_mappings = {
        'contacts': 'contacts',
        'calendar': 'calendar',
        'email': 'email',
        'messages': 'messages',
        'notes': 'notes',
        'llm_convos': 'conversations',
        'conversations': 'conversations',
        'health_sleep': 'health_sleep',
        'health_activities': 'health_activities',
        'health_nutrition': 'health_nutrition',
        'sleep': 'health_sleep',
        'activities': 'health_activities',
        'nutrition': 'health_nutrition',
    }
    
    # Check for each keyword
    for keyword, json_key in type_mappings.items():
        if keyword in filename_lower:
            return json_key
    
    # If no match, return cleaned filename (remove extension and special chars)
    base_name = filename.replace('.csv', '').lower()
    # Extract last part after dash if present
    if ' - ' in base_name:
        base_name = base_name.split(' - ')[-1]
    # Clean up
    base_name = re.sub(r'[^a-z0-9_]', '_', base_name)
    return base_name

# File uploader
st.subheader("Upload Artifact CSV Files")
st.write("Upload all persona artifact CSV files. The app will automatically detect the type from the filename.")

artifact_files = st.file_uploader(
    "Upload CSV files",
    type=['csv'],
    accept_multiple_files=True,
    key='artifacts',
    help="Files like 'Name - Contacts.csv', 'Name - Calendar.csv', etc."
)

if artifact_files:
    st.write(f"**Uploaded {len(artifact_files)} files**")
    
    # Show what was detected
    detected_types = {}
    for file in artifact_files:
        artifact_type = detect_artifact_type(file.name)
        detected_types[file.name] = artifact_type
    
    st.write("**Detected artifact types:**")
    for filename, detected_type in detected_types.items():
        st.write(f"- `{filename}` → **{detected_type}**")
    
    if st.button("Generate JSON", type="primary"):
        try:
            # Build the persona JSON
            persona_json = {}
            
            # Process each uploaded file
            for file in artifact_files:
                artifact_type = detect_artifact_type(file.name)
                
                st.write(f"Processing: {file.name} → {artifact_type}")
                
                # Read and clean the CSV
                artifact_df = pd.read_csv(file)
                cleaned_df = clean_dataframe(artifact_df)
                
                # Convert to list of records
                persona_json[artifact_type] = cleaned_df.to_dict(orient='records')
                
                st.success(f"✓ Loaded {artifact_type}: {artifact_df.shape[0]} rows, {artifact_df.shape[1]} columns")
            
            # Convert to JSON string
            json_str = json.dumps(persona_json, default=json_serialize, indent=2)
            
            st.write(f"Debug - JSON string length: {len(json_str):,} characters")
            
            st.success(f"✓ Generated persona JSON with {len(persona_json)} artifact types!")
            
            # Show approximate token count (JSON is ~3.5 chars per token due to structure)
            char_count = len(json_str)
            estimated_tokens = int(char_count / 3.5)
            st.metric(label="Estimated Tokens (approximate)", value=f"{estimated_tokens:,}")
            st.caption(f"Character count: {char_count:,} | Note: Estimate may vary by ±5-10%")
            
            # Preview
            with st.expander("Preview JSON structure"):
                st.json({k: f"{len(v)} records" for k, v in persona_json.items()})
            
            # Download button
            st.download_button(
                label="📥 Download Persona JSON",
                data=json_str,
                file_name="persona_data.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.write("Please check that your CSV files are valid and properly formatted.")

st.markdown("---")
st.markdown("**Instructions:**")
st.markdown("""
1. Upload all artifact CSV files at once
2. The app automatically detects the artifact type from the filename
3. Click 'Generate JSON'
4. Download your consolidated persona JSON file

**Supported artifact types:**
- Contacts, Calendar, Email, Messages, Notes
- LLM_Convos (becomes "conversations")
- Health_sleep, Health_activities, Health_nutrition

**Example filenames:**
- `Jordan Reid - Contacts.csv` → "contacts"
- `Jordan Reid (template copy) - Calendar.csv` → "calendar"
- `Person Name - LLM_Convos.csv` → "conversations"
""")
