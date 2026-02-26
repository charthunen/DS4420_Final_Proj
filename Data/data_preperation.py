import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import json
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

np.random.seed(42)

# load the tmdb datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# merge on movie id
data = movies.merge(credits, left_on='id', right_on='movie_id', suffixes=('', '_c'))
data.head()

def parse_json(val):
    try: return json.loads(val) if pd.notna(val) else []
    except: return []

data['genres_list'] = data['genres'].apply(parse_json)
data['cast_list'] = data['cast'].apply(parse_json)
data['crew_list'] = data['crew'].apply(parse_json)
data['companies_list'] = data['production_companies'].apply(parse_json)
data['keywords_list'] = data['keywords'].apply(parse_json)

# genre one-hot encoding
for g in ['Action','Adventure','Animation','Comedy','Crime','Drama',
          'Family','Fantasy','Horror','Romance','Science Fiction','Thriller']:
    col = f"genre_{g.lower().replace(' ','_')}"
    data[col] = data['genres_list'].apply(lambda gs: int(any(x['name']==g for x in gs)))

data['num_genres'] = data['genres_list'].apply(len)
data['num_cast'] = data['cast_list'].apply(len)

# get director name
data['director'] = data['crew_list'].apply(
    lambda c: next((m['name'] for m in c if m.get('job')=='Director'), 'Unknown'))

# major studio flag
majors = {'Warner Bros.','Universal Pictures','Paramount Pictures',
           'Walt Disney Pictures','Columbia Pictures','20th Century Fox',
           'New Line Cinema','Lionsgate','DreamWorks','Sony Pictures'}
data['is_major_studio'] = data['companies_list'].apply(
    lambda cs: int(any(c['name'] in majors for c in cs)))
data['num_companies'] = data['companies_list'].apply(len)

# release date features
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year
data['release_month'] = data['release_date'].dt.month
data['is_summer'] = data['release_month'].isin([5,6,7]).astype(int)
data['is_holiday'] = data['release_month'].isin([11,12]).astype(int)

# franchise indicator from keywords
sequel_kw = {'sequel','based on novel','based on comic','superhero','marvel comic','dc comics'}
data['is_franchise'] = data['keywords_list'].apply(
    lambda ks: int(any(k['name'].lower() in sequel_kw for k in ks)))

data['is_english'] = (data['original_language']=='en').astype(int)

# log transforms
data['log_budget'] = np.where(data['budget']>0, np.log1p(data['budget']), 0)
data['log_revenue'] = np.where(data['revenue']>0, np.log1p(data['revenue']), 0)

# filter to movies with valid budget, revenue, post-2000, valid runtime
data_clean = data[(data['budget']>1000) & (data['revenue']>0) &
                  (data['release_year']>=2000) & (data['runtime']>0)].copy()
print(f"{len(data_clean)} movies after cleaning")

# select features and target 
feature_cols = [
    'log_budget','runtime','popularity','vote_average','vote_count',
    'num_genres','num_cast','is_major_studio','num_companies',
    'is_summer','is_holiday','is_franchise','is_english',
    'genre_action','genre_adventure','genre_animation','genre_comedy',
    'genre_crime','genre_drama','genre_family','genre_fantasy',
    'genre_horror','genre_romance','genre_science_fiction','genre_thriller'
]

Phi = data_clean[feature_cols].to_numpy()
y = data_clean['log_revenue'].to_numpy()

# scaling 

scaler = MinMaxScaler()
Phi_scale = scaler.fit_transform(Phi)

# train/test split 

[Phi_train, Phi_test, y_train, y_test] = train_test_split(Phi_scale, y, test_size=0.2)

print(f"Train: {Phi_train.shape[0]}, Test: {Phi_test.shape[0]}")
print(f"Features: {len(feature_cols)}")

#  export for R 

export_df = data_clean[['id','title'] + feature_cols + ['log_revenue']].reset_index(drop=True)
export_df.to_csv("data/movies_clean.csv", index=False)
print("Saved data/movies_clean.csv")

# still need to:
# - use pytrends to pull 4-week pre-release search interest
# - twitter (use tweepy or a Kaggle dataset for pre-release mention volume)
