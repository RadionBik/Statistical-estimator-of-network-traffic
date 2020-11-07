import pandas as pd
from .parsed_fields import ParsedFields as PF


def form_df(stats: dict) -> pd.DataFrame:
    stats = pd.DataFrame(stats)
    # calc IAT and convert from seconds to ms
    iat_from = stats[stats[PF.is_source]][PF.ts].diff().fillna(0) * 1000
    iat_to = stats[~stats[PF.is_source]][PF.ts].diff().fillna(0) * 1000

    stats[PF.iat] = pd.concat([iat_to, iat_from]).sort_index().round(0)
    return stats.reset_index(drop=True)
