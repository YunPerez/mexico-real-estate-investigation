###################################################
# script to create competitors data set
# @roman 2/11/24
###################################################
# %% S0: Libraries
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import os

from scipy.spatial import KDTree
from scipy.special import logsumexp

# %% S0.1: Parameters
GEO_WEIGHT = 0.3
TERRAIN_WEIGHT = 0.1
CONSERVATION_WEIGHT = 0.15
BUILT_WEIGHT = 0.20
CHAR_WEIGHT = 0.15
TIME_WEIGHT = 0.1
CITY = 10
BATCH_SIZE_COMPS = 500
RADIUS_COMPS = 2.5
COL_FOR_STATS = 'log_price_per_sqm'
DISTANCE_PENALIZATION = 2
BATCH_SIZE_INFO = 2_000
MAX_NUM_COMPS = 60

SAVE_DIR = "../../data/tmp"
COLS2STAY = [
    # variables to stay
    'observation_id',
    # geographic
    'longitude', 'latitude',
    # topology
    'land_area', 'built_area',
    # characteristics
    'elevator_service_id_cat', 'max_total_levels_recat', 'age_in_months',
    'bedrooms_cat', 'full_bathrooms_cat', 'parking_lots_cat',
    'half_bathrooms_cat', 'property_class_id_cat',
    'conservation_status_id_recat',
    # time
    'valuation_date',
]


# %%  S0.2: Functions
# Read data
def get_properties(city=None):
    # read properties
    gdf = pd.read_parquet("../../data/interim/cleaned_data_s4.parquet")

    # subset if city is given
    if city is not None:
        gdf = gdf[gdf['city_cluster'] == city]

    # to geopandas
    gdf = gpd.GeoDataFrame(
        gdf,
        geometry=gpd.points_from_xy(
            gdf['longitude'], gdf['latitude']
        ),
        crs="EPSG:4326"
        )

    # change crs to 6372
    gdf = gdf.to_crs("EPSG:6372")

    return gdf


def read_cities():
    return (
        gpd
        .read_parquet("../../data/misc/cities.parquet")
        .to_crs("EPSG:6372")
        ['cluster'].unique()
        )


def wrangle_gdf(gdf):
    # important columns
    general_cols_to_stay = [
        # variables to stay
        'observation_id',
        # price
        'log_price_per_sqm', 'price_per_sqm',
        # geographic
        'longitude', 'latitude',
        # topology
        'land_area', 'built_area',
        # characteristics
        'elevator_service_id_cat', 'max_total_levels_recat', 'age_in_months',
        'bedrooms_cat', 'full_bathrooms_cat', 'parking_lots_cat',
        'half_bathrooms_cat', 'property_class_id_cat',
        'conservation_status_id_recat',
        # time
        'valuation_date',
        # filtering
        'city_cluster', 'property_type',
    ]

    # wrangle data
    gdf = (
        gdf
        .copy()
        # new vars
        .assign(
            property_type=lambda x: np.where(
                x['property_type_id'].le(3), 'house', 'apartment'
                ),
            longitude=lambda x: x["geometry"].x,
            latitude=lambda x: x["geometry"].y,
            parking_lots=lambda x: x["parking_lots"].fillna(0),
            price_per_sqm=lambda x: x["price"] / x["saleable_area"],
            log_price_per_sqm=lambda x: np.log(x["price_per_sqm"]),
        )
        .reset_index(drop=True)
        # categorize variables for comparisson
        .assign(
            bedrooms_cat=lambda x: np.select(
                [
                    x["bedrooms"].le(1),
                    x["bedrooms"].le(2),
                    x["bedrooms"].le(3),
                    x["bedrooms"].gt(3),
                ],
                [
                    1, 2, 3, 4
                ],
                default=0
            ),
            full_bathrooms_cat=lambda x: np.select(
                [
                    x["full_bathrooms"].le(1),
                    x["full_bathrooms"].le(2),
                    x["full_bathrooms"].le(3),
                    x["full_bathrooms"].gt(3),
                ],
                [
                    1, 2, 3, 4
                ],
                default=0
            ),
            half_bathrooms_cat=lambda x: np.select(
                [
                    x["half_bathrooms"].le(1),
                    x["half_bathrooms"].le(2),
                    x["half_bathrooms"].le(3),
                    x["half_bathrooms"].gt(3),
                ],
                [
                    1, 2, 3, 4
                ],
                default=0
            ),
            property_class_id_cat=lambda x: np.select(
                [
                    x["property_class_id"].le(3),
                    x["property_class_id"].le(4),
                    x["property_class_id"].gt(4),
                ],
                [
                    1, 2, 3
                ],
                default=0
            ),
            conservation_status_id_recat=lambda x:
                x["conservation_status_id"].replace({7: 4.5}),
            max_total_levels=lambda x:
                x[['level', 'total_levels']].max(axis=1),
            max_total_levels_recat=lambda x: np.select(
                [
                    x["max_total_levels"].le(1),
                    x["max_total_levels"].le(5),
                    x["max_total_levels"].le(15),
                    x["max_total_levels"].gt(15),
                ],
                [
                    1, 2, 3, 4
                ],
                default=0
            ),
            elevator_service_id_cat=lambda x:
                np.where(x['elevator_service_id'].eq(1), 1, 0),
            parking_lots_cat=lambda x: np.select(
                [
                    x["parking_lots"].le(0),
                    x["parking_lots"].le(1),
                    x["parking_lots"].le(2),
                    x["parking_lots"].gt(2),
                ],
                [
                    0, 1, 2, 3
                ],
                default=0
            ),
        )
        .loc[:, general_cols_to_stay]
    )

    return gdf


# Competitors
def min_max(x):
    return (x - x.min()) / (x.max() - x.min())


def get_neighbors_properties(gdf, r=1):
    # copy
    gdf = gdf.copy()

    # fit kdtree
    kdtree = KDTree(
        data=gdf[['longitude', 'latitude']],
    )

    # get neighbors at r-km
    return kdtree.query_ball_point(
        gdf[['longitude', 'latitude']],
        r=r * 1_000,
        workers=-1
    )


def get_possible_neighbors(df_own, df_theirs, vars_list):
    # Select only relevant columns and explode neighbors_list column in df_own
    df_own_expanded = (
        df_own
        .loc[:, vars_list + ['neighbors_list']]
        .explode('neighbors_list')  # Expands neighbors_list into multiple rows
        .rename(columns={'neighbors_list': 'id_neighbor'})  # Rename 4 clarity
        .reset_index()  # Reset index to use in merging
    )

    # Prepare df_theirs for merging by keeping track of original indexes
    df_theirs_indexed = (
        df_theirs
        .assign(index=lambda x: x.index)  # Add index column for merging
        .loc[:, ['index'] + vars_list]  # Select only relevant columns
    )

    # Merge expanded df_own with df_theirs based on neighbor IDs
    merged_df = df_own_expanded.merge(
        df_theirs_indexed,
        how='inner',
        left_on='id_neighbor',  # Match on the id_neighbor column from df_own
        right_on='index',  # Match on index from df_theirs
        suffixes=('_own', '_neighbor')
    )

    # Filter out rows where observation_id_own is the
    # same as observation_id_neighbor
    result_df = merged_df.query(
        "observation_id_own != observation_id_neighbor", engine='python'
        )

    return result_df


def distance_of_competitors(gdf):
    # Step 1: Distance
    return (
        gdf.copy()
        # helpers
        .assign(
            # time distance
            time_distance=lambda x: np.sqrt(
                ((
                    x['valuation_date_own'] - x['valuation_date_neighbor']
                    ).dt.days / (365 * 2))**2
            ),
            land_area_proportion=lambda x:
                x['land_area_neighbor'] / x['land_area_own'],
            built_area_proportion=lambda x:
                x['built_area_neighbor'] / x['built_area_own'],
        )
        # filter
        .query("time_distance.le(1) & land_area_proportion.between(0.5, 2) & built_area_proportion.between(0.25, 4)", engine='python')
        .drop(
            columns=[
                'time_distance',
                'land_area_proportion',
                'built_area_proportion'
            ]
        )
        # distances: TODO: Check distributions
        .assign(
            # geo distance
            geo_distance=lambda x: np.sqrt(
                (x['longitude_own'] - x['longitude_neighbor'])**2
                + (x['latitude_own'] - x['latitude_neighbor'])**2
            ),
            # topology distance
            terrain_distance=lambda x: np.sqrt(
                (x['land_area_neighbor'] - x['land_area_own'])**2
            ),
            built_distance=lambda x: np.sqrt(
                (x['built_area_neighbor'] - x['built_area_own'])**2
            ),
            # conservation
            conservation_distance=lambda x: np.sqrt(
                (x['age_in_months_own'] - x['age_in_months_neighbor'])**2
                + (
                    x['conservation_status_id_recat_own'] 
                    - x['conservation_status_id_recat_neighbor']
                    )**2
            ),
            # characteristics distance
            characteristics_distance=lambda x: np.sqrt(
                (x['elevator_service_id_cat_own']
                    - x['elevator_service_id_cat_neighbor'])**2
                + (x['max_total_levels_recat_own']
                    - x['max_total_levels_recat_neighbor'])**2
                + (x['bedrooms_cat_own']
                    - x['bedrooms_cat_neighbor'])**2
                + (x['full_bathrooms_cat_own']
                    - x['full_bathrooms_cat_neighbor'])**2
                + (x['half_bathrooms_cat_own']
                    - x['half_bathrooms_cat_neighbor'])**2
                + (x['parking_lots_cat_own']
                    - x['parking_lots_cat_neighbor'])**2
                + (x['property_class_id_cat_own']
                    - x['property_class_id_cat_neighbor'])**2
            ),
            # time distance
            time_distance=lambda x: np.sqrt(
                # TODO: think in months
                ((
                    x['valuation_date_own']
                    - x['valuation_date_neighbor']
                 ).dt.days / (365 * 2))**2
            ),
        )
        .assign(
            # normalize distances 
            geo_distance_norm=lambda x:
                min_max(x['geo_distance']),
            terrain_distance_norm=lambda x:
                min_max(x['terrain_distance']),
            built_distance_norm=lambda x:
                min_max(x['built_distance']),
            conservation_distance_norm=lambda x:
                min_max(x['conservation_distance']),
            characteristics_distance_norm=lambda x:
                min_max(x['characteristics_distance']),
            time_distance_norm=lambda x:
                min_max(x['time_distance']),
            # total distance
            total_distance=lambda x: (
                x['geo_distance_norm'] * GEO_WEIGHT
                + x['terrain_distance_norm'] * TERRAIN_WEIGHT
                + x['built_distance_norm'] * BUILT_WEIGHT
                + x['conservation_distance_norm'] * CONSERVATION_WEIGHT
                + x['characteristics_distance_norm'] * CHAR_WEIGHT
                + x['time_distance_norm'] * TIME_WEIGHT
                )
        )
        # eliminate 0s in the distance to avoid comparing to itself
        .query("total_distance.gt(0)")
        .assign(
            total_distance=lambda x:
                (
                    x
                    .groupby('observation_id_own')
                    ['total_distance']
                    .transform(min_max)
                    )
            )  # TODO: CHANGE FOR OBS_ID
        .sort_values(by=['observation_id_own', 'total_distance'])
        # top n competitors
        .groupby('observation_id_own', as_index=False)
        .head(MAX_NUM_COMPS)
        .reset_index(drop=True)
    )


def subset_competitors(gdf):
    # Aggregation
    return (
        gdf
        .assign(
            index_distance_tuple=lambda x: list(
                zip(x['observation_id_neighbor'], x['total_distance'])
                ),
        )
        .groupby('observation_id_own', as_index=False)
        .agg(
            neighbors_list=('index_distance_tuple', list),
            num_neighbors=('observation_id_neighbor', 'count'),
        )
    )


def get_competitors(
        gdf, cols_to_stay, prop_type='apartment',
        batch_size=None, radius=1
     ):
    # subset
    gdf_work = (
        gdf
        .query(
            "property_type.eq(@prop_type)",
            engine='python'
            )
        .reset_index(drop=True)
        .copy()
    )

    # check if there are properties
    if gdf_work.shape[0] == 0:
        print("No properties to compare")
        return pd.DataFrame()

    # get neighbors
    gdf_work['neighbors_list'] = get_neighbors_properties(gdf_work, r=radius)

    # get possible neighbors by batch
    n_batches = gdf_work.shape[0] // batch_size
    if n_batches > 0:
        batch_indexes = np.array_split(
            gdf_work.index, gdf_work.shape[0] // batch_size
            )
    else:
        batch_indexes = [gdf_work.index]

    gdf_neighbors_list = []
    for batch in tqdm(batch_indexes, desc="Processing chunks"):
        # get possible neighbors
        gdf_neighbors_info = get_possible_neighbors(
            df_own=gdf_work.loc[batch],
            df_theirs=gdf_work,
            vars_list=cols_to_stay
        )
        # find competitors & append
        gdf_neighbors_list.append(
                subset_competitors(distance_of_competitors(gdf_neighbors_info))
            )

    # concatenate
    gdf_neighbors = pd.concat(gdf_neighbors_list)

    return gdf_neighbors


# Statistics
def competitors_stats(df, col_to_summarize, xi=0.2):
    """
    Calculate weighted statistics using numerically stable computations.

    Args:
        df: DataFrame containing the data
        col_to_summarize: Column name to compute statistics for
        xi: Scale parameter for distance weighting

    Returns:
        Series containing weighted and unweighted statistics
    """
    n_neighbors = df.shape[0]

    # Calculate log weights to avoid overflow/underflow
    log_weights_unorm = -xi * np.sqrt(n_neighbors) * df['total_distance']

    # Use logsumexp trick for normalization
    log_normalizer = logsumexp(log_weights_unorm)
    log_weights = log_weights_unorm - log_normalizer

    # Convert back to weights space when needed
    weights = np.exp(log_weights)
    sq_sum_weights = np.sum(weights**2) if n_neighbors > 1 else np.nan

    # Get array to summarize
    x_array = df[col_to_summarize].values

    # Compute weighted mean
    w_mean = np.sum(x_array * weights)

    # Compute statistics
    return pd.Series({
        'weighted_mean': w_mean,
        'weighted_std': np.sqrt(
            # unbiased weighted std (theorem)
            (1/(1 - sq_sum_weights)) * np.sum(weights * (x_array - w_mean)**2)
        ),
        'mean': np.mean(x_array),
        'std': np.std(x_array),
        'num_neighbors': n_neighbors,
    })


def get_info_from_competitors(
    df, df_info, col_to_summarize, xi=0.2, batch_size=2_000
     ):
    # Initialize results list
    all_results = []

    # Get unique combinations of city and property type
    partitions = (
        df_info['property_type']
        .drop_duplicates()
        .values
    )

    for prop_type in partitions:
        print(f"Processing data for property type {prop_type}")
        # Filter relevant data
        df_partition = df[
            (df['property_type'] == prop_type)
        ].copy()

        df_info_partition = df_info[
            (df_info['property_type'] == prop_type)
        ].copy()

        if df_partition.empty or df_info_partition.empty:
            print(f"No data for property type {prop_type}")
            continue

        # Process partition in batches
        chunks = np.array_split(
            df_partition, max(1, df_partition.shape[0] // batch_size)
            )
        merge_cols = ['observation_id', col_to_summarize]

        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Expand chunk
            exploded_chunk = (
                chunk
                .explode('neighbors_list')
                .reset_index(drop=True)
                .assign(
                    observation_id_neighbor=lambda x:
                        x['neighbors_list'].str[0],
                    total_distance=lambda x:
                        x['neighbors_list'].str[1]
                )
                .merge(
                    df_info_partition[merge_cols],
                    left_on='observation_id_neighbor',
                    right_on='observation_id',
                    suffixes=('_own', '_neighbor')
                )
                .drop(columns=['observation_id', 'neighbors_list'])
            )

            # Calculate statistics
            df_competitors_info = (
                exploded_chunk
                .groupby('observation_id_own', as_index=False)
                .apply(
                    competitors_stats,
                    col_to_summarize=col_to_summarize,
                    xi=xi,
                    include_groups=False
                )
            )

            all_results.append(df_competitors_info)

    # Combine all results
    if not all_results:
        print("No data to process")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# Save
def save_dataframe(df, file_name, city):
    os.makedirs(SAVE_DIR, exist_ok=True)
    df.to_parquet(
        f"{SAVE_DIR}/{file_name}_{city}.parquet",
        index=False
    )


# main
def main_comps(
    city, batch_comps, radius_comps_to_search,
    batch_info, distance_penalizer, cols2stay,
    col4stats
     ):
    # Part 1: Read data
    print(f"{'='*10}Comps for city {city}{'='*10}")
    print("Reading data of ")
    gdf_properties = get_properties(city=city)
    print(f"num props: {gdf_properties.shape[0]}")

    # Part 2: Wrangle data
    print("Wrangling data")
    gdf_properties = wrangle_gdf(gdf_properties)
    print(f"num props: {gdf_properties.shape[0]}")

    # Part 3: Get competitors
    # calculate competitors
    print("Calculating competitors")
    property_types_list = gdf_properties['property_type'].unique()

    df_competitors_list = []
    for ptype in property_types_list:
        # get competitors
        df_comps = get_competitors(
            gdf_properties,
            cols2stay,
            prop_type=ptype,
            batch_size=batch_comps,
            radius=radius_comps_to_search  # search radius in km
            )
        # append
        df_competitors_list.append(df_comps)

    # concatenate
    df_competitors = pd.concat(df_competitors_list)
    print(f"num competitors found: {df_competitors.shape[0]}")

    # Part 4: Get comp info
    print("Getting competitors info")
    # add city and property type for each observation_id
    df_competitors = (
        df_competitors
        .merge(
            gdf_properties[
                ['observation_id', 'city_cluster', 'property_type']
                ],
            left_on='observation_id_own',
            right_on='observation_id',
            how='inner'
        )
        .drop(columns='observation_id')
    )

    # get info from competitors
    df_competitors_info = get_info_from_competitors(
        df=df_competitors,
        df_info=gdf_properties,
        col_to_summarize=col4stats,
        xi=distance_penalizer,
        batch_size=batch_info
        )

    #  Part 5: save
    print("Saving data")
    # save competitors
    save_dataframe(
        (
            df_competitors
            .loc[:, ['observation_id_own', 'neighbors_list']]
            .astype(str)
            ),
        "competitors",
        city
        )
    # save competitors info
    save_dataframe(df_competitors_info, "competitors_info", city)


# main
# read cities
cities = read_cities()
# cities = [25]

for city in cities:
    main_comps(
        city=city,
        batch_comps=BATCH_SIZE_COMPS,
        radius_comps_to_search=RADIUS_COMPS,
        batch_info=BATCH_SIZE_INFO,
        distance_penalizer=DISTANCE_PENALIZATION,
        cols2stay=COLS2STAY,
        col4stats=COL_FOR_STATS
    )

print("Done! Bye ...")
