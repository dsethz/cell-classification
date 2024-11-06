import pandas as pd
from nucleus_3d_classification.baseline_models.model_fit import remove_diff_columns

def test_remove_diff_columns():
    # Create sample dataframes
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

    df2 = pd.DataFrame({
        'A': [10, 11, 12],
        'B': [13, 14, 15],
        'D': [16, 17, 18]
    })

    df3 = pd.DataFrame({
        'A': [19, 20, 21],
        'B': [22, 23, 24],
        'E': [25, 26, 27]
    })

    # Expected result after removing different columns
    expected_df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    expected_df2 = pd.DataFrame({
        'A': [10, 11, 12],
        'B': [13, 14, 15]
    })

    expected_df3 = pd.DataFrame({
        'A': [19, 20, 21],
        'B': [22, 23, 24]
    })

    # Apply the function
    result_df1, result_df2, result_df3 = remove_diff_columns(df1, df2, df3)

    # Check if the result matches the expected dataframes
    pd.testing.assert_frame_equal(result_df1, expected_df1)
    pd.testing.assert_frame_equal(result_df2, expected_df2)
    pd.testing.assert_frame_equal(result_df3, expected_df3)

if __name__ == "__main__":
    test_remove_diff_columns()
    print("All tests passed.")