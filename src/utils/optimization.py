from numpy import finfo, iinfo, int8, int16, int32, int64, float16, float32, float64


def reduce_mem_usage(df, verbose=False):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > iinfo(int8).min and c_max < iinfo(int8).max:
                    df[col] = df[col].astype(int8)
                elif c_min > iinfo(int16).min and c_max < iinfo(int16).max:
                    df[col] = df[col].astype(int16)
                elif c_min > iinfo(int32).min and c_max < iinfo(int32).max:
                    df[col] = df[col].astype(int32)
                elif c_min > iinfo(int64).min and c_max < iinfo(int64).max:
                    df[col] = df[col].astype(int64)
            else:
                if c_min > finfo(float16).min and c_max < finfo(float16).max:
                    df[col] = df[col].astype(float16)
                elif c_min > finfo(float32).min and c_max < finfo(float32).max:
                    df[col] = df[col].astype(float32)
                else:
                    df[col] = df[col].astype(float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df
