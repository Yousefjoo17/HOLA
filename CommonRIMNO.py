# Get unique RIMNO sets
prime_rimnos = set(prime_df['RIMNO'].dropna().unique())
transaction_rimnos = set(transaction_df['RIMNO'].dropna().unique())

# Find intersection
common_rimnos = sorted(prime_rimnos.intersection(transaction_rimnos))

# Print count
print(f"Number of common RIMNOs: {len(common_rimnos)}\n")

# Print values (be careful if very large)
print("Common RIMNOs:")
for rim in common_rimnos:
    print(rim)

# Save to TXT file
output_file = "Common RIMNO.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Number of common RIMNOs: {len(common_rimnos)}\n\n")
    f.write("Common RIMNOs:\n")
    
    for rim in common_rimnos:
        f.write(f"{rim}\n")

print(f"\nSaved to '{output_file}' successfully.")