import pandas as pd
import numpy as np



from . logging import logger


def _validate_names(kind, atlas_name, agg_function):
    if '$' in kind:
        raise ValueError("Feature kind must not have the special character $")
    if '$' in atlas_name:
        raise ValueError("Atlas name must not have the special character $")
    if '$' in agg_function:
        raise ValueError(
            "Agg function mamemust not have the special character $")


def _to_table_name(kind, atlas_name, agg_function):
    _validate_names(kind, atlas_name, agg_function)
    table_name = f'{kind}${atlas_name}'
    if agg_function is not None:
        table_name = f'{table_name}${agg_function}'
    return table_name


def _from_table_name(table_name):
    splitted = table_name.split('$')
    kind = splitted[0]
    atlas_name = splitted[1]
    agg_function = splitted[2]
    return kind, atlas_name, agg_function


def list_features_old(uri):
    """List features from a SQL Database

    Parameters
    ----------
    uri : str
        The connection URI.
        Easy options:
            'sqlite://' for an in memory sqlite database
            'sqlite:///<path_to_file>' to save in a file

        Check https://docs.sqlalchemy.org/en/14/core/engines.html for more
        options

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the features list
    """
    logger.debug(f'Listing features from DB {uri}')
    engine = create_engine(uri, echo=False)
    features = {'kind': [], 'atlas_name': [], 'agg_function': []}
    for t_name in inspect(engine).get_table_names():
        t_k, t_a, t_f = _from_table_name(t_name)
        features['kind'].append(t_k)
        features['atlas_name'].append(t_a)
        features['agg_function'].append(t_f)
    return pd.DataFrame(features)


def list_features(uri):
    storage = SQLiteFeatureStorage(uri, single_output=True)
    return storage.list_features()


def read_features_old(uri, kind, atlas_name, index_col, agg_function=None):
    """Read features from a SQL Database

    Parameters
    ----------
    uri : str
        The connection URI.
        Easy options:
            'sqlite://' for an in memory sqlite database
            'sqlite:///<path_to_file>' to save in a file

        Check https://docs.sqlalchemy.org/en/14/core/engines.html for more
        options
    kind : str
        kind of features
    altas_name : str
        the name of the atlas
    index_col : list(str)
        The columns to be used as index
    agg_function : str
        The aggregation function used (defaults to None)

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the features
    """
    from sqlalchemy import create_engine, inspect
    table_name = _to_table_name(kind, atlas_name, agg_function)
    logger.debug(f'Reading data from DB {uri} - table {table_name}')
    engine = create_engine(uri, echo=False)
    df = pd.read_sql(table_name, con=engine, index_col=index_col)
    return df


def read_features(uri, feature_name):
    from junifer.storage import SQLiteFeatureStorage
    storage = SQLiteFeatureStorage(uri, single_output=True)
    df = storage.read_df(feature_name=feature_name)
    df = df.xs(0, level=1, drop_level=True)
    df.index = [f'sub-{x}' for x in df.index]
    df.index.name = 'SubjectID'
    return df


def read_prs(fname):
    data = np.genfromtxt(fname=fname, delimiter="\t")
    subj_id = data[0].astype(int)
    prs = data[1]
    vfunc = np.vectorize(lambda x: f'sub-{x}')
    subj_id = vfunc(subj_id)
    df = pd.DataFrame({'SubjectID': subj_id, 'prs': prs})
    return df.set_index('SubjectID').dropna()


def read_pheno(fname):
    df = pd.read_csv(fname, sep=',')
    df['SubjectID'] = [f'sub-{x}' for x in df['eid']]  # type: ignore

    df.drop(columns={'eid'}, inplace=True)  # type: ignore

    return df.set_index('SubjectID')  # type: ignore


def read_apoe(fname):
    data = np.genfromtxt(fname=fname, delimiter="\t", dtype=str)
    subj_id = data[0, 1:].astype(int)
    vfunc = np.vectorize(lambda x: f'sub-{x}')
    subj_id = vfunc(subj_id)
    data_dict = {'SubjectID': subj_id}
    for col in data[1:]:
        col_name = col[0]
        col_data = col[1:]
        data_dict[col_name] = col_data

    df = pd.DataFrame(data_dict).set_index('SubjectID')
    df['APOE'] = df['rs429358'] + df['rs7412']
    return df


def read_features_jay(features_path):
    import datatable
    from datatable import dt, fread, f
    datatable.options.nthreads = 1
    features = {
        "1_gmd_schaefer_all_subjects.jay": "GMD_Schaefer1000x7",
        "2_gmd_SUIT_all_subjects.jay": "GMD_SUIT",
        "4_gmd_tian_all_subjects.jay": "GMD_Tian",
        "dk_gray_white_contrast.jay": "Surface_GrayWhiteContrast_dk",
        "dk_pial_surface.jay": "Surface_PialSurface_dk",
        "dk_white_surface.jay": "Surface_WhiteSurface_dk",
        "dk_white_thickness.jay": "Surface_WhiteThickness_dk",
        "dk_white_volume.jay": "Surface_WhiteVolume_dk",
        "fALFF_Schaefer1000x7_Mean.jay": "fALFF_Schaefer1000x7",
        "fALFF_SUIT_Mean.jay": "fALFF_SUIT",
        "fALFF_Tian_Mean.jay": "fALFF_Tian",
        "GCOR_Schaefer1000x7_Mean.jay": "GCOR_Schaefer1000x7",
        "GCOR_SUIT_Mean.jay": "GCOR_SUIT",
        "GCOR_Tian_Mean.jay": "GCOR_Tian",
        "LCOR_Schaefer1000x7_Mean.jay": "LCOR_Schaefer1000x7",
        "LCOR_SUIT_Mean.jay": "LCOR_SUIT",
        "LCOR_Tian_Mean.jay": "LCOR_Tian",
    }


    final_dt = None
    for fname, prefix in features.items():
        logger.info(f"Reading features from {fname}")
        t_dt = fread(features_path / fname)
        t_dt.key = "SubjectID"
        t_dt.names = {
            col: f"{prefix}_{col.replace('(', '').replace(')', '')}"
            for col in t_dt.names if col != "SubjectID"
        }
        logger.info(f"\tSamples: {t_dt.shape[0]}")
        logger.info(f"\tFeatures: {t_dt.shape[1] - 1}")
        if final_dt is None:
            final_dt = t_dt
        else:
            final_dt = final_dt[:, :, dt.join(t_dt)]
    logger.info("Filtering out samples with missing features")
    final_dt = final_dt[dt.rowall(f[:] != None), :]
    logger.info(f"\tSamples: {final_dt.shape[0]}")
    logger.info(f"\tFeatures: {final_dt.shape[1] - 1}")
    logger.info("Converting to pandas")
    final_df = final_dt.to_pandas()
    logger.info("Reading done")
    return final_df.set_index('SubjectID')
