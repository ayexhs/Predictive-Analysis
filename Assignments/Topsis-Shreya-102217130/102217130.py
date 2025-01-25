import sys
import pandas as pd
import numpy as np

def validate_input(weights, impacts, num_criteria):
    if len(weights) != num_criteria or len(impacts) != num_criteria:
        raise ValueError("Number of weights and impacts must match the number of criteria.")
    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Impacts should be either '+' or '-'.")
    weights = [float(w) for w in weights]
    return weights

def topsis(input_file, weights, impacts, result_file):
    try:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')  # Read input file with encoding
        if df.shape[1] < 3:
            raise ValueError("Input file must contain at least three columns.")

        # Extracting numerical values, ignoring the first column (Fund Names)
        criteria_data = df.iloc[:, 1:].values.astype(float)
        num_cols = criteria_data.shape[1]

        weights = validate_input(weights, impacts, num_cols)

        # Normalize the decision matrix
        norm_matrix = criteria_data / np.sqrt((criteria_data ** 2).sum(axis=0))

        # Apply weights to the normalized matrix
        weighted_matrix = norm_matrix * weights

        # Determine ideal best and worst solutions
        ideal_best = np.where(np.array(impacts) == '+', weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
        ideal_worst = np.where(np.array(impacts) == '+', weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

        # Calculate Euclidean distance from ideal best and worst
        distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
        distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

        # Compute TOPSIS score
        topsis_score = distance_worst / (distance_best + distance_worst)

        # Add Topsis Score and Rank to DataFrame
        df['Topsis Score'] = topsis_score
        df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)

        # Save the result to CSV
        df.to_csv(result_file, index=False)
        print(f"TOPSIS result saved to {result_file}")

    except FileNotFoundError:
        print("Error: Input file not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFile>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2].split(',')
    impacts = sys.argv[3].split(',')
    result_file = sys.argv[4]

    topsis(input_file, weights, impacts, result_file)
