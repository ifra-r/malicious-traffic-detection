# Checks the class balance of the classification column.

# If the class distribution falls outside the 45–55% range, it downsamples the larger class to restore balance.

# Saves the balanced dataset as a new file.

import pandas as pd

def check_and_balance_class(csv_path, target_column, output_path="data/3_balanced_data.csv", tolerance=5):
    print(f"\n🔍 Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.\n")

        # Check if target column exists
        if target_column not in df.columns:
            print(f"[❌ Error] Column '{target_column}' not found in the dataset.")
            return

        print(f"🎯 Target column: '{target_column}'")

        # Get class counts and percentages (before)
        class_counts = df[target_column].value_counts()
        class_percentages = df[target_column].value_counts(normalize=True) * 100

        print("\n📊 Class distribution BEFORE balancing:")
        for label in class_counts.index:
            print(f"  Class '{label}': {class_counts[label]} samples ({class_percentages[label]:.2f}%)")

        # Imbalance check
        max_percent = class_percentages.max()
        min_percent = class_percentages.min()
        diff = max_percent - min_percent

        if diff > (2 * tolerance):
            print(f"\n⚠️ Imbalance detected: {diff:.2f}% difference exceeds {2 * tolerance}% threshold")
            print("📉 Proceeding to downsample the majority class...")

            # Get majority and minority classes
            min_class = class_percentages.idxmin()
            max_class = class_percentages.idxmax()

            df_min = df[df[target_column] == min_class]
            df_max = df[df[target_column] == max_class].sample(n=len(df_min), random_state=42)

            # Combine and shuffle
            df_balanced = pd.concat([df_min, df_max]).sample(frac=1, random_state=42).reset_index(drop=True)

            # Save balanced data
            df_balanced.to_csv(output_path, index=False)
            print(f"\n💾 Balanced dataset saved to: {output_path}")

            # Report after balancing
            final_counts = df_balanced[target_column].value_counts()
            final_percentages = df_balanced[target_column].value_counts(normalize=True) * 100

            print("\n📊 Class distribution AFTER balancing:")
            for label in final_counts.index:
                print(f"  Class '{label}': {final_counts[label]} samples ({final_percentages[label]:.2f}%)")

            print("\n✅ Dataset is now balanced and ready for use.")

        else:
            print(f"\n✅ Class distribution is within {tolerance}% tolerance. No balancing needed.")
            print("ℹ️ No changes made to the dataset.")

    except FileNotFoundError:
        print(f"[❌ Error] File '{csv_path}' not found.")
    except Exception as e:
        print(f"[❌ Error] {str(e)}")


# === Example Usage ===
csv_file = "data/2_missing_vals.csv"
target_col = "classification"
check_and_balance_class(csv_file, target_col)