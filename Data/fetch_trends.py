import pandas as pd
import numpy as np
import time
from datetime import timedelta
from pytrends.request import TrendReq  # type: ignore

# load cleaned movie list
data = pd.read_csv('data/movies_clean.csv')
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data = data.dropna(subset=['release_date'])

pytrends = TrendReq(hl='en-US', tz=360)

def get_prerelease_interest(title, release_date):
    end_date = release_date
    start_date = release_date - timedelta(days=28)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    try:
        pytrends.build_payload([title], cat=0, timeframe=timeframe, geo='US', gprop='')
        df = pytrends.interest_over_time()
        if df.empty or title not in df.columns:
            return 0.0
        return float(df[title].mean())
    except Exception:
        return np.nan

# check for existing cache to resume if interrupted
try:
    cache = pd.read_csv('data/trends_cache.csv')
    done_ids = set(cache['id'].tolist())
    results = cache.to_dict('records')
    print(f"Resuming from cache, {len(done_ids)} already done")
except FileNotFoundError:
    done_ids = set()
    results = []

# scrape one movie at a time with delay to avoid rate limits
for i, row in data.iterrows():
    if row['id'] in done_ids:
        continue

    interest = get_prerelease_interest(row['title'], row['release_date'])
    results.append({'id': row['id'], 'google_trends_interest': interest})

    # save cache every 10 movies in case of interruption
    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv('data/trends_cache.csv', index=False)
        print(f"  {len(results)}/{len(data)} done, last: {row['title']} = {interest:.1f}")

    # sleep to avoid rate limiting
    time.sleep(2)

# final save
pd.DataFrame(results).to_csv('data/trends_cache.csv', index=False)
print(f"Done. Saved {len(results)} rows to data/trends_cache.csv")
