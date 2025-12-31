import pandas as pd               # Pandas for data manipulation
import tqdm                       # Progress bar for chunk processing
import numpy as np                # Numerical operations

# CSV mapping field IDs to human-readable names
field_map_file = "field_ids_enriched.csv"
field_dict = {}                             # Map original field ID -> new column name
tabular_fields = []                         # List of tabular feature column names
with open(field_map_file, "r", encoding="utf-8") as f:  # Open the field mapping file
    next(f)  # skip header line
    for line in f:  # Iterate through lines
        parts = line.strip().split(",")  # Split by CSV commas
        if len(parts) >= 3:               # Ensure we have at least id and name columns (fix: was >=2)
            # Original field identifier (e.g., "34-0.0")
            field_id = parts[0]
            field_name = parts[2]         # Human-readable column name
            field_dict[field_id] = field_name  # Record the mapping
            # Track as a potential tabular feature
            tabular_fields.append(field_name)
# Exclude raw date parts and target columns
exclude_fields = ['year', 'month', 'Death', 'age_at_assessment']
tabular_fields = [
    # Filter out excluded columns
    field for field in tabular_fields if field not in exclude_fields]

# TSV mapping field IDs to ICD10-related date columns
field_to_icd_map = "icd10_codes_mod.tsv"
# Date-like variables to be converted to offsets
date_vars = []
with open(field_to_icd_map, "r", encoding="utf-8") as f:  # Open ICD10 mapping
    for line in f:  # Iterate each mapping row
        parts = line.strip().split()  # Split on whitespace for TSV
        if len(parts) >= 6:           # Guard against malformed lines
            # Map field ID to the date column name
            field_dict[parts[0]] = parts[5]
            date_vars.append(parts[5])       # Track date column names in order

for j in range(17):                        # Map up to 17 cancer entry slots (dates and types)
    # Cancer diagnosis date slot j
    field_dict[f'40005-{j}.0'] = f'cancer_date_{j}'
    field_dict[f'40006-{j}.0'] = f'cancer_type_{j}'  # Cancer type/code slot j

# Number of ICD-related date columns before adding extras
len_icd = len(date_vars)
date_vars.extend(['Death', 'date_of_assessment'] +  # Add outcome date and assessment date
                 # Add cancer date columns
                 [f'cancer_date_{j}' for j in range(17)])

labels_file = "labels.csv"  # File listing label codes
label_dict = {}              # Map code string -> integer label id
with open(labels_file, "r", encoding="utf-8") as f:  # Open labels file
    for idx, line in enumerate(f):  # Enumerate to assign incremental label IDs
        parts = line.strip().split(' ')  # Split by space
        if parts and parts[0]:           # Guard against empty lines
            # Map code to index (0 for padding, 1 for checkup)
            label_dict[parts[0]] = idx + 2

event_list = []  # Accumulator for event arrays across chunks
tabular_list = []  # Accumulator for tabular feature DataFrames across chunks
ukb_iterator = pd.read_csv(  # Stream UK Biobank data in chunks
    "ukb_data.csv",
    sep=',',
    chunksize=10000,          # Stream file in manageable chunks to reduce memory footprint
    # First column (participant ID) becomes DataFrame index
    index_col=0,
    low_memory=False         # Disable type inference optimization for consistent dtypes
)
# Iterate chunks with progress
for ukb_chunk in tqdm.tqdm(ukb_iterator, desc="Processing UK Biobank data"):
    # Rename columns to friendly names
    ukb_chunk = ukb_chunk.rename(columns=field_dict)
    # Require sex to be present
    ukb_chunk.dropna(subset=['sex'], inplace=True)

    # Construct date of birth from year and month (day fixed to 1)
    ukb_chunk['day'] = 1
    ukb_chunk['dob'] = pd.to_datetime(
        # Guard against malformed dates
        ukb_chunk[['year', 'month', 'day']], errors='coerce'
    )
    del ukb_chunk['day']

    # Use only date variables that actually exist in the current chunk
    present_date_vars = [c for c in date_vars if c in ukb_chunk.columns]

    # Convert date-like columns to datetime and compute day offsets from dob
    if present_date_vars:
        date_cols = ukb_chunk[present_date_vars].apply(
            pd.to_datetime, format="%Y-%m-%d", errors='coerce'  # Parse dates safely
        )
        date_cols_days = date_cols.sub(
            ukb_chunk['dob'], axis=0)   # Timedelta relative to dob
        ukb_chunk[present_date_vars] = date_cols_days.apply(
            lambda x: x.dt.days)  # Store days since dob

    # Append tabular features (use only columns that exist)
    present_tabular_fields = [
        c for c in tabular_fields if c in ukb_chunk.columns]
    tabular_list.append(ukb_chunk[present_tabular_fields].copy())

    # Process disease events from ICD10-related date columns
    # Take ICD date cols plus 'Death' if present by order
    icd10_cols = present_date_vars[:len_icd + 1]
    # Melt to long form: participant id, event code (column name), and days offset
    melted_df = ukb_chunk.reset_index().melt(
        id_vars=['eid'],
        value_vars=icd10_cols,
        var_name='event_code',
        value_name='days',
    )
    # Require non-missing day offsets
    melted_df.dropna(subset=['days'], inplace=True)
    if not melted_df.empty:
        melted_df['label'] = melted_df['event_code'].map(
            label_dict)  # Map event code to numeric label
        # Fix: ensure labels exist before int cast
        melted_df.dropna(subset=['label'], inplace=True)
        if not melted_df.empty:
            event_list.append(
                melted_df[['eid', 'days', 'label']]
                .astype(int)  # Safe now since label and days are non-null
                .to_numpy()
            )

    # Optimized cancer processing without wide_to_long
    cancer_frames = []
    for j in range(17):
        d_col = f'cancer_date_{j}'
        t_col = f'cancer_type_{j}'
        if d_col in ukb_chunk.columns and t_col in ukb_chunk.columns:
            # Filter rows where both date and type are present
            mask = ukb_chunk[d_col].notna() & ukb_chunk[t_col].notna()
            if mask.any():
                subset_idx = ukb_chunk.index[mask]
                subset_days = ukb_chunk.loc[mask, d_col]
                subset_type = ukb_chunk.loc[mask, t_col]

                # Map cancer type to label
                # Use first 3 chars
                cancer_codes = subset_type.str.slice(0, 3)
                labels = cancer_codes.map(label_dict)

                # Filter valid labels
                valid_label_mask = labels.notna()
                if valid_label_mask.any():
                    # Create array: eid, days, label
                    # Ensure types are correct for numpy
                    c_eids = subset_idx[valid_label_mask].values
                    c_days = subset_days[valid_label_mask].values
                    c_labels = labels[valid_label_mask].values

                    # Stack
                    chunk_cancer_data = np.column_stack(
                        (c_eids, c_days, c_labels))
                    cancer_frames.append(chunk_cancer_data)

    if cancer_frames:
        event_list.append(np.vstack(cancer_frames))

# Combine tabular chunks
final_tabular = pd.concat(tabular_list, axis=0, ignore_index=False)
final_tabular.index.name = 'eid'  # Ensure index named consistently
data = np.vstack(event_list)      # Stack all event arrays into one

# Sort by participant then day
data = data[np.lexsort((data[:, 1], data[:, 0]))]

# Keep only events with non-negative day offsets
data = data[data[:, 1] >= 0]

# Remove duplicate (participant_id, label) pairs keeping first occurrence.
data = pd.DataFrame(data).drop_duplicates([0, 2]).values

# Store compactly using unsigned 32-bit integers
data = data.astype(np.uint32)

# Select eid in both data and tabular
valid_eids = np.intersect1d(data[:, 0], final_tabular.index)
data = data[np.isin(data[:, 0], valid_eids)]
final_tabular = final_tabular.loc[valid_eids]
final_tabular = final_tabular.convert_dtypes()

# Save [eid, sex, date_of_assessment] for basic info
basic_info = final_tabular[['sex', 'date_of_assessment']]
basic_info.to_csv("ukb_basic_info.csv")

# Drop sex and date_of_assessment from tabular features
final_tabular = final_tabular.drop(columns=['sex', 'date_of_assessment'])

# Process categorical columns in tabular features
# If a column is integer type with few unique values, treat as categorical. For each integer column:
# Count unique values (exclude NaN, and negative values if any) as C, set NaN or negative to 0, remap original values to [1..C].
for col in final_tabular.select_dtypes(include=['Int64', 'int64']).columns:
    # Get unique values efficiently
    series = final_tabular[col]
    unique_vals = series.dropna().unique()

    # Filter negatives from unique values
    valid_vals = sorted([v for v in unique_vals if v >= 0])

    if len(valid_vals) <= 10:  # Threshold for categorical
        # Create mapping
        val_map = {val: idx + 1 for idx, val in enumerate(valid_vals)}

        # Map values. Values not in val_map (negatives, NaNs) become NaN
        mapped_col = series.map(val_map)

        # Fill NaN with 0 and convert to uint32
        final_tabular[col] = mapped_col.fillna(0).astype(np.uint32)

# Save processed tabular features
final_tabular.to_csv("ukb_table.csv")

# Save event data
np.save("ukb_event_data.npy", data)
