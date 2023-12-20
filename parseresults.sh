#!/bin/bash

# Values for c and k
c_values=(2 3 10)
k_values=(1 4 8 16 32)

# Iterate over c and k values
for c in "${c_values[@]}"; do
  for k in "${k_values[@]}"; do
    # Construct the path based on c and k
    path="out/randomrj/${c}/${k}"

    # Run the Python script with parameters
    python3 results.py out/randomgt/random.csv "$path" out/randomres/ --c "$c" --k "$k" 
  done
done
