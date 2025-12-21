import streamlit as st
import pandas as pd
import json
import numpy as np
import re
import tiktoken

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
    """Clean DataFrame for JSON compatibility and remove week column and ID columns"""
    # Remove week column if present
    if 'week' in df.columns:
        df = df.drop('week', axis=1)
    
    # Remove all ID columns (ending with _id or named 'id')
    id_columns = [col for col in df.columns if col.endswith('_id') or col.lower() == 'id']
    if id_columns:
        df = df.drop(id_columns, axis=1)
    
    # Fix phone numbers and similar columns that became floats BEFORE replacing NaN
    for col in df.columns:
        if any(x in col.lower() for x in ['phone', 'contact', 'number']):
            # Convert to string and clean, but only for non-null values
            df[col] = df[col].apply(
                lambda x: str(int(x)) if pd.notna(x) and x != '' else None
            )
    
    # Now replace any remaining NaN values with None
    df_clean = df.replace({np.nan: None})
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'bool':
            df_clean[col] = df_clean[col].map({True: True, False: False, None: None})
    
    return df_clean

def convert_to_compact_array(df):
    """Convert DataFrame to compact array format - removing null values"""
    df_clean = clean_dataframe(df)
    # Convert to list of dicts, removing null values from each record
    return [
        {k: v for k, v in row.items() if v is not None}
        for row in df_clean.to_dict(orient='records')
    ]

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
                
                # Read and convert to compact array
                artifact_df = pd.read_csv(file)
                persona_json[artifact_type] = convert_to_compact_array(artifact_df)
                
                st.success(f"✓ Loaded {artifact_type}: {artifact_df.shape[0]} rows, {artifact_df.shape[1]} columns")
            
            # Convert to JSON string - NO INDENTATION for compact output
            json_str = json.dumps(persona_json, default=json_serialize, ensure_ascii=False)
            
            st.success(f"✓ Generated persona JSON with {len(persona_json)} artifact types!")
            
            # Show character count
            char_count = len(json_str)
            st.write(f"**Character count:** {char_count:,}")
            
            # Calculate estimated tokens (approximation method)
            estimated_tokens = int(char_count / 3.5)
            
            # Calculate exact token count with tiktoken
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(json_str)
                exact_tokens = len(tokens)
                
                # Show both side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Estimated Tokens (÷3.5)", value=f"{estimated_tokens:,}")
                with col2:
                    st.metric(label="Exact Tokens (tiktoken)", value=f"{exact_tokens:,}")
                
                # Show difference
                difference = exact_tokens - estimated_tokens
                st.caption(f"Difference: {difference:,} tokens ({abs(difference/exact_tokens*100):.1f}%)")
                
            except Exception as e:
                st.error(f"Could not calculate exact tokens: {e}")
                st.metric(label="Estimated Tokens (÷3.5)", value=f"{estimated_tokens:,}")
                st.caption("Character count ÷ 3.5 (approximation)")
            
            # Preview
            with st.expander("Preview JSON structure"):
                st.json({k: f"{len(v)} records" for k, v in persona_json.items()})
            
            # Download buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download JSON File",
                    data=json_str,
                    file_name="persona_data.json",
                    mime="application/json",
                    help="Download as .json file"
                )
            
            with col2:
                st.download_button(
                    label="📋 Download as Text (for copy/paste)",
                    data=json_str,
                    file_name="persona_data.txt",
                    mime="text/plain",
                    help="Download as .txt file with UTF-8 encoding for easy copy/paste into templates"
                )
            
            # Add a text area with the JSON for direct copying
            with st.expander("📋 Copy JSON directly from here"):
                st.text_area(
                    "JSON Content (click to select all, then copy)",
                    json_str,
                    height=300,
                    help="Select all text (Cmd+A / Ctrl+A) and copy (Cmd+C / Ctrl+C)"
                )
            
            # Add verification info
            try:
                st.caption(f"File size: {len(json_str.encode('utf-8')):,} bytes | When you download this file, it should have {exact_tokens:,} tokens when tested locally.")
            except:
                st.caption(f"File size: {len(json_str.encode('utf-8')):,} bytes")
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.write("Please check that your CSV files are valid and properly formatted.")

st.markdown("---")
st.markdown("**Instructions:**")
st.markdown("""
1. Upload all artifact CSV files at once
2. The app automatically detects the artifact type from the filename
3. Click 'Generate JSON'
4. Choose your download option:
   - **Download JSON File**: Standard .json file
   - **Download as Text**: .txt file with UTF-8 encoding for easy copy/paste
   - **Copy directly**: Use the expandable text area to copy directly

**Supported artifact types:**
- Contacts, Calendar, Email, Messages, Notes
- LLM_Convos (becomes "conversations")
- Health_sleep, Health_activities, Health_nutrition

**Example filenames:**
- `Jordan Reid - Contacts.csv` → "contacts"
- `Jordan Reid (template copy) - Calendar.csv` → "calendar"
- `Person Name - LLM_Convos.csv` → "conversations"
""")
