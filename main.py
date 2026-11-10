"""
Project 06: Data Wrangling Practice
====================================
This script demonstrates comprehensive data wrangling techniques including
data cleaning, handling missing values, data transformation, formatting,
and normalization using pandas.

Original notebooks: 17 notebooks combined

Author: Data Analysis Portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

# ================================================================================
# DATA LOADING
# ================================================================================

def load_auto_data():
    """Load the automobile dataset."""
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
    
    headers = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
        "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
        "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
        "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
    ]
    
    print(f"Loading data from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), names=headers)
        print("Data loaded successfully from URL!")
    except Exception as e:
        print(f"Failed to load from URL: {e}")
        print("Creating sample data for demonstration...")
        np.random.seed(42)
        n = 205
        df = pd.DataFrame({
            'symboling': np.random.randint(-2, 3, n),
            'normalized-losses': np.random.choice([str(i) for i in range(65, 256)] + ['?'], n),
            'make': np.random.choice(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury', 'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'], n),
            'fuel-type': np.random.choice(['gas', 'diesel'], n),
            'aspiration': np.random.choice(['std', 'turbo'], n),
            'num-of-doors': np.random.choice(['two', 'four', '?'], n),
            'body-style': np.random.choice(['convertible', 'hardtop', 'hatchback', 'sedan', 'wagon'], n),
            'drive-wheels': np.random.choice(['4wd', 'fwd', 'rwd'], n),
            'engine-location': np.random.choice(['front', 'rear'], n),
            'wheel-base': np.random.uniform(86, 121, n).round(1),
            'length': np.random.uniform(141, 208, n).round(1),
            'width': np.random.uniform(60, 72, n).round(1),
            'height': np.random.uniform(47, 60, n).round(1),
            'curb-weight': np.random.randint(1488, 4066, n),
            'engine-type': np.random.choice(['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'], n),
            'num-of-cylinders': np.random.choice(['eight', 'five', 'four', 'six', 'three', 'twelve', 'two'], n),
            'engine-size': np.random.randint(61, 326, n),
            'fuel-system': np.random.choice(['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi'], n),
            'bore': np.random.choice([str(round(x, 2)) for x in np.arange(2.5, 4.0, 0.01)] + ['?'], n),
            'stroke': np.random.choice([str(round(x, 2)) for x in np.arange(2.0, 4.2, 0.01)] + ['?'], n),
            'compression-ratio': np.random.uniform(7, 23, n).round(2),
            'horsepower': np.random.choice([str(i) for i in range(48, 289)] + ['?'], n),
            'peak-rpm': np.random.choice([str(i) for i in range(4150, 6600)] + ['?'], n),
            'city-mpg': np.random.randint(13, 49, n),
            'highway-mpg': np.random.randint(16, 54, n),
            'price': np.random.choice([str(i) for i in range(5118, 45400)] + ['?'], n),
        })
    
    return df

# ================================================================================
# DATA WRANGLING FUNCTIONS
# ================================================================================

def explore_data(df):
    """Initial data exploration."""
    print("\n" + "=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print(f"\nBasic statistics:\n{df.describe(include='all')}")
    print(f"\nMissing values ('?'):")
    for col in df.columns:
        count = (df[col] == '?').sum()
        if count > 0:
            print(f"  {col}: {count}")

def clean_missing_values(df):
    """Handle missing values marked as '?'."""
    print("\n" + "=" * 50)
    print("HANDLING MISSING VALUES")
    print("=" * 50)
    
    df_clean = df.copy()
    
    # Replace '?' with NaN
    df_clean.replace('?', np.nan, inplace=True)
    print("Replaced '?' with NaN")
    
    # Check missing data
    missing = df_clean.isnull().sum()
    print(f"\nMissing values per column:\n{missing[missing > 0]}")
    
    # Convert numeric columns
    numeric_cols = [
        'normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price'
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Fill missing values with mean for continuous variables
    fill_mean_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']
    for col in fill_mean_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mean_val = df_clean[col].mean()
            df_clean[col].fillna(mean_val, inplace=True)
            print(f"Filled {col} with mean: {mean_val:.2f}")
    
    # Fill price with mean (or drop rows)
    if 'price' in df_clean.columns and df_clean['price'].isnull().any():
        # Drop rows where price is missing (target variable)
        df_clean.dropna(subset=['price'], inplace=True)
        print(f"Dropped rows with missing price. Remaining: {len(df_clean)}")
    
    # Fill categorical with mode
    if 'num-of-doors' in df_clean.columns and df_clean['num-of-doors'].isnull().any():
        mode_val = df_clean['num-of-doors'].mode()[0]
        df_clean['num-of-doors'].fillna(mode_val, inplace=True)
        print(f"Filled num-of-doors with mode: {mode_val}")
    
    print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum()}")
    
    return df_clean

def data_formatting(df):
    """Format and standardize data."""
    print("\n" + "=" * 50)
    print("DATA FORMATTING")
    print("=" * 50)
    
    df_fmt = df.copy()
    
    # Convert mpg to L/100km
    if 'city-mpg' in df_fmt.columns:
        df_fmt['city-L/100km'] = 235 / df_fmt['city-mpg']
        print("Converted city-mpg to city-L/100km")
    
    if 'highway-mpg' in df_fmt.columns:
        df_fmt['highway-L/100km'] = 235 / df_fmt['highway-mpg']
        print("Converted highway-mpg to highway-L/100km")
    
    # Normalize dimensions (mm to inches if needed, or just standardize)
    # Convert length, width, height from inches to cm
    dim_cols = ['length', 'width', 'height', 'wheel-base']
    for col in dim_cols:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col] * 2.54  # inches to cm
    print("Converted dimensions from inches to cm")
    
    # Normalize curb-weight (lbs to kg)
    if 'curb-weight' in df_fmt.columns:
        df_fmt['curb-weight-kg'] = df_fmt['curb-weight'] * 0.453592
        print("Converted curb-weight from lbs to kg")
    
    # Engine size (cubic inches to liters)
    if 'engine-size' in df_fmt.columns:
        df_fmt['engine-size-L'] = df_fmt['engine-size'] * 0.0163871
        print("Converted engine-size to liters")
    
    return df_fmt

def data_normalization(df):
    """Normalize numeric features."""
    print("\n" + "=" * 50)
    print("DATA NORMALIZATION")
    print("=" * 50)
    
    df_norm = df.copy()
    
    # Select numeric columns for normalization
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude columns that shouldn't be normalized
    exclude = ['symboling', 'compression-ratio', 'city-mpg', 'highway-mpg']
    norm_cols = [c for c in numeric_cols if c not in exclude]
    
    print(f"Normalizing columns: {norm_cols}")
    
    # Min-Max Normalization
    for col in norm_cols:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
    
    # Z-score Normalization
    for col in norm_cols:
        if col in df_norm.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[f'{col}_zscore'] = (df_norm[col] - mean_val) / std_val
    
    print(f"Added normalized columns. New shape: {df_norm.shape}")
    
    # Visualize normalization effect
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sample_cols = [c for c in norm_cols[:3] if c in df_norm.columns]
    
    for i, col in enumerate(sample_cols):
        # Original
        axes[0, i].hist(df_norm[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'Original: {col}')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')
        
        # Min-Max normalized
        norm_col = f'{col}_norm'
        if norm_col in df_norm.columns:
            axes[1, i].hist(df_norm[norm_col].dropna(), bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1, i].set_title(f'Min-Max Normalized: {col}')
            axes[1, i].set_xlabel('Normalized Value')
            axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved normalization_comparison.png")
    plt.close()
    
    return df_norm

def binning_data(df):
    """Bin continuous variables into categories."""
    print("\n" + "=" * 50)
    print("DATA BINNING")
    print("=" * 50)
    
    df_bin = df.copy()
    
    # Bin horsepower
    if 'horsepower' in df_bin.columns:
        bins = [0, 100, 150, 200, 300]
        labels = ['Low', 'Medium', 'High', 'Very High']
        df_bin['horsepower_bin'] = pd.cut(df_bin['horsepower'], bins=bins, labels=labels, include_lowest=True)
        print(f"Horsepower binning:\n{df_bin['horsepower_bin'].value_counts()}")
    
    # Bin price
    if 'price' in df_bin.columns:
        bins = [0, 10000, 20000, 30000, 50000]
        labels = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
        df_bin['price_bin'] = pd.cut(df_bin['price'], bins=bins, labels=labels, include_lowest=True)
        print(f"Price binning:\n{df_bin['price_bin'].value_counts()}")
    
    # Bin engine size
    if 'engine-size' in df_bin.columns:
        df_bin['engine_size_bin'] = pd.qcut(df_bin['engine-size'], q=4, labels=['Small', 'Medium', 'Large', 'Very Large'])
        print(f"Engine size binning:\n{df_bin['engine_size_bin'].value_counts()}")
    
    return df_bin

def indicator_variables(df):
    """Create indicator (dummy) variables for categorical data."""
    print("\n" + "=" * 50)
    print("INDICATOR VARIABLES")
    print("=" * 50)
    
    df_ind = df.copy()
    
    # Select categorical columns
    cat_cols = df_ind.select_dtypes(include=['object']).columns.tolist()
    # Exclude columns with too many unique values
    cat_cols = [c for c in cat_cols if df_ind[c].nunique() < 15]
    
    print(f"Creating dummies for: {cat_cols}")
    
    # Create dummy variables
    df_dummies = pd.get_dummies(df_ind[cat_cols], prefix=cat_cols, drop_first=True)
    
    # Combine with original (excluding original categorical columns)
    df_result = pd.concat([df_ind.drop(columns=cat_cols), df_dummies], axis=1)
    
    print(f"Original shape: {df_ind.shape}, New shape: {df_result.shape}")
    print(f"New columns: {df_dummies.columns.tolist()[:10]}...")
    
    return df_result

def correlation_analysis(df):
    """Perform correlation analysis."""
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Correlation matrix
    corr_matrix = numeric_df.corr()
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Top correlations with price
    if 'price' in corr_matrix.columns:
        price_corr = corr_matrix['price'].sort_values(key=abs, ascending=False)
        print(f"\nTop correlations with price:")
        print(price_corr.head(15))
    
    # Visualize correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved correlation_heatmap.png")
    plt.close()
    
    return corr_matrix

# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 06: DATA WRANGLING PRACTICE")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading automobile data...")
    df = load_auto_data()
    
    # Explore
    print("\n[2] Exploring data...")
    explore_data(df)
    
    # Clean missing values
    print("\n[3] Cleaning missing values...")
    df = clean_missing_values(df)
    
    # Format data
    print("\n[4] Formatting data...")
    df = data_formatting(df)
    
    # Normalize
    print("\n[5] Normalizing data...")
    df = data_normalization(df)
    
    # Binning
    print("\n[6] Binning data...")
    df = binning_data(df)
    
    # Indicator variables
    print("\n[7] Creating indicator variables...")
    df = indicator_variables(df)
    
    # Correlation analysis
    print("\n[8] Correlation analysis...")
    correlation_analysis(df)
    
    print("\n" + "=" * 60)
    print("DATA WRANGLING COMPLETE")
    print("=" * 60)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:20]}...")
