import pandas as pd
from pandas.api.types import is_numeric_dtype
import pytest

from modelling.data_processing.data_processing import connect_to_s3, load_concat_df, convert_variable_type, calculate_time_variables, clean_flat_model, categorise_stories, convert_to_title_case, split_dataset, print_missing_counts, output_to_s3

@pytest.fixture
def sample_raw_df():
    """
    A pytest fixture that returns a sample Pandas DataFrame.
    """
    sample_raw_df = pd.DataFrame({"month":["2023-08"],
                                  "town":["ANG MO KIO"],
                                  "flat_type":["2 ROOM"],
                                  "block":["156"],
                                  "street_name":["ANG MO KIO AVE 3"],
                                  "floor_area_sqm":["44.0"],
                                  "flat_model":["Improved"],
                                  "lease_commence_date":["1984"],
                                  "storey_range":["06 TO 10"],
                                  "resale_price":["250000"],})
    return sample_raw_df

def test_connect_to_s3():
    try: 
        bucket = connect_to_s3()
        assert bucket.name == 'hdb-resale-pred-raw', "Bucket name mismatch"
        
    except Exception as e:
        assert False, f"connect_to_s3() raised an exception: {e}"

def test_load_concat_df():
    input_bucket = connect_to_s3()
    
    try:
        df_all = load_concat_df(input_bucket)
        assert isinstance(df_all, pd.DataFrame), "Output is not a DataFrame"
        assert not df_all.empty, "Loaded DataFrame is empty"
        
    except Exception as e:
        assert False, f"load_concat_df() raised an exception: {e}"
        
        
def test_convert_variable_type(sample_raw_df):
    assert isinstance(sample_raw_df, pd.DataFrame), "Input is not a DataFrame"
    
    proc_df = convert_variable_type(sample_raw_df)
    assert is_numeric_dtype(proc_df["resale_price"]), "resale_price not converted to numeric"
    assert is_numeric_dtype(proc_df["lease_commence_date"]), "lease_commence_date not converted to numeric"
    assert is_numeric_dtype(proc_df["floor_area_sqm"]), "floor_area_sqm not converted to numeric"
    assert proc_df["month"].dtype.str == "<M8[ns]", "month not converted to datetime"
    
def test_calculate_time_variables(sample_raw_df):
    proc_df = convert_variable_type(sample_raw_df)
    proc_df = calculate_time_variables(proc_df)
    assert 'flat_age_years' in proc_df.columns, "flat_age_years column missing"
    assert 'days_from_earliest_data' in proc_df.columns, "days_from_earliest_data column missing"
    assert proc_df['flat_age_years'].iloc[0] == 39, "Incorrect flat_age_years calculation"
    assert proc_df['days_from_earliest_data'].iloc[0] == 4170, "days_from_earliest_data should be positive"
    
def test_clean_flat_model(sample_raw_df):
    proc_df = clean_flat_model(sample_raw_df)
    assert "flat_model_revised" in proc_df.columns, "flat_model_revised column missing"
    assert proc_df['flat_model_revised'].iloc[0] == 'Improved', "flat_model_revised not cleaned correctly for non-Maisonette"
    
    revised_proc_df = sample_raw_df.copy()
    revised_proc_df.loc[0, 'flat_model'] = 'Maisonette XYZ'
    revised_proc_df = clean_flat_model(revised_proc_df)
    assert revised_proc_df['flat_model_revised'].iloc[0] == 'Maisonette', "flat_model_revised not cleaned correctly for Maisonette"
    
def test_categorise_stories(sample_raw_df):
    proc_df = categorise_stories(sample_raw_df)
    assert 'storey_range_grouped' in proc_df.columns, "storey_range_grouped column missing"
    assert proc_df['storey_range_grouped'].iloc[0] == '1-15', "Incorrect storey_range_grouped categorization"
    assert proc_df['storey_max'].iloc[0] == 10, "Incorrect storey_max extraction"
    
def test_convert_to_title_case(sample_raw_df):
    proc_df = clean_flat_model(sample_raw_df)
    proc_df = convert_to_title_case(proc_df)
    assert proc_df['town'].iloc[0] == 'Ang Mo Kio', "town not converted to title case"
    assert proc_df['flat_type'].iloc[0] == '2 Room', "flat_type not converted to title case"
    assert proc_df['flat_model_revised'].iloc[0] == 'Improved', "flat_model_revised not converted to title case"
    
