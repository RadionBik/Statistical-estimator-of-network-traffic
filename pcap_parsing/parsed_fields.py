import pandas as pd


class ParsedFields:
    ps = 'PS, байт'
    iat = 'IAT, мс'
    ts = 'TS'
    is_source = 'От источника'


def select_features(df: pd.DataFrame):
    return df.loc[:, (ParsedFields.ps, ParsedFields.iat)].values, df[ParsedFields.is_source].values
