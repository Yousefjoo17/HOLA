temp_df['DOB'] = pd.to_numeric(temp_df['DOB'], errors='coerce')

temp_df.loc[temp_df['DOB'] < 0, 'DOB'] = np.nan

temp_df['DOB'] = pd.to_datetime(
    temp_df['DOB'],
    origin='1899-12-30',
    unit='D',
    errors='coerce'
)
