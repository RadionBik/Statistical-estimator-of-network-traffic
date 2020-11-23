import logging

import pandas as pd
import seaborn as sns

from settings import RANDOM_SEED

logger = logging.getLogger(__name__)


def auto_select_model_by_bic(
        model_class,
        df,
        columns: tuple = None,
        n_comp=None,
        min_comp=5,
        max_comp=20,
        step_comp=1,
        return_bic_dict=False,
) -> tuple:
    if not n_comp:
        comp_range = range(min_comp, max_comp + 1, step_comp)
        logger.info(f'Started fitting model with auto number of components')
    else:
        comp_range = range(n_comp, n_comp + 1)
        logger.info(f'Started fitting model')

    comp_model = {}
    comp_bic = {}
    for comp in comp_range:
        model = model_class(n_components=comp,
                            covariance_type="full",
                            random_state=RANDOM_SEED)
        df = df[columns] if columns else df

        model.fit(df)
        comp_model[comp] = model
        comp_bic[comp] = model.bic(df)
        logger.info(f'fit model with {comp} components, BIC={comp_bic[comp]}')

    best_comp, best_bic = min(comp_bic.items(), key=lambda x: x[1])
    best_model = comp_model[best_comp]
    logger.info(f'Best BIC={best_bic} is with {best_comp} components')

    if return_bic_dict:
        return best_model, comp_bic
    return best_model,


def plot_bics(bics: dict, direction):
    bics_df = pd.DataFrame(bics.items())
    bics_df.columns = ['Число компонент', 'BIC']
    ax = sns.regplot(data=bics_df, x='Число компонент', y='BIC',
                     order=2, label=direction, scatter=True)
    ax.legend()
