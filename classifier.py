import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle as pkl
from geopy.geocoders import Nominatim
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm

def create_dataset(dset='train'):
	dataset = pd.read_csv('./data/{0}.csv'.format(dset))
	tour_info = pd.read_csv('./data/tours.csv')
	tour_info.rename({'biker_id':'host_id', 'latitude':'tour_latitude', 'longitude':'tour_longitude'}, axis=1, inplace=True)
	biker_info = pd.read_csv('./data/bikers.csv')

	dataset = dataset.merge(right=biker_info, how='left', on='biker_id')
	dataset = dataset.merge(right=tour_info, how='left', on='tour_id')
	
	timestamp_hour = []
	timestamp_min = []
	timestamp_sec = []
	timestamp_year = []
	timestamp_month = []
	timestamp_day = []
	biker_time_zone = []
	tour_day = []
	tour_month = []
	tour_year = []
	biker_bornIn = []
	biker_member_since_day = []
	biker_member_since_month = []
	biker_member_since_year = []
	tour_latitude = []
	tour_longitude = []
	tour_time_zone = []
	age_while_tour = []
	num_friends_going = []
	num_friends_not_going = []
	num_friends_maybe = []
	num_friends_invited = []
	tour_going = []
	tour_not_going = []
	tour_maybe = []
	perc_friends_going = []
	perc_friends_not_going = []
	perc_friends_maybe = []
	perc_friends_invited = []
	is_host_friend = []
	member_how_long_days = []
	
	
	format = '%d-%m-%Y %H:%M:%S'
	for index in dataset.index:
		date_time = dataset.iloc[index]['timestamp']
		dt = datetime.strptime(date_time, format)
		timestamp_year.append(dt.year)
		timestamp_month.append(dt.month)
		timestamp_day.append(dt.day)
		timestamp_hour.append(dt.hour)
		timestamp_min.append(dt.minute)
		timestamp_sec.append(dt.second)
		
		
	for index in dataset.index:
		biker_id = dataset.iloc[index]['biker_id']
		time = dataset.iloc[index]['time_zone']
		area = dataset.iloc[index]['area']
		location_id = dataset.iloc[index]['location_id']
		if np.isnan(time):
		    if area == 'Epsom':
		        biker_time_zone.append(60)
		    elif 'Yogyakarta' in area:
		        biker_time_zone.append(420)
		    elif 'Los Angeles' in area:
		        biker_time_zone.append(-480)
		    elif 'Abuja' in area:
		        biker_time_zone.append(60)
		    elif 'London' in area:
		        biker_time_zone.append(60)
		    elif 'Sigli  Aceh' in area:
		        biker_time_zone.append(420)
		    elif 'San Francisco' in area:
		        biker_time_zone.append(-420)
		    elif 'Liverpool' in area:
		        biker_time_zone.append(240)
		    elif location_id == 'FR':
		        biker_time_zone.append(60)
		    elif location_id == 'US':
		        biker_time_zone.append(-240)
		else:
		    biker_time_zone.append(time)
		    
		    
	locator = Nominatim(user_agent='myGeocoder')
	format = '%d-%m-%Y'
	for index in dataset.index:
		date = dataset.iloc[index]['tour_date']
		cty = dataset.iloc[index]['city']
		stt = dataset.iloc[index]['state']
		ctry = dataset.iloc[index]['country']
		ltt = dataset.iloc[index]['tour_latitude']
		lgt = dataset.iloc[index]['tour_longitude']
		mdate = dataset.iloc[index]['member_since']
		byear = dataset.iloc[index]['bornIn']

		address = ''
		if isinstance(cty, str):
		    address += (cty + ',')
		if isinstance(stt, str):
		    address += (stt + ',')
		if isinstance(ctry, str):
		    address += ctry

		if np.isnan(ltt) or np.isnan(lgt):
		    if address:
		        try:
		            loc = locator.geocode(address)
		            tour_latitude.append(loc.latitude)
		            tour_longitude.append(loc.longitude)
		            tour_time_zone.append(4*loc.longitude)
		        except:
		            tour_latitude.append(16.5)
		            tour_longitude.append(25.5)
		            tour_time_zone.append(102)

		    else:
		        tour_latitude.append(16.5)
		        tour_longitude.append(25.5)
		        tour_time_zone.append(102)
		else:
		    tour_latitude.append(ltt)
		    tour_longitude.append(lgt)
		    tour_time_zone.append(4*lgt)

		if byear == 'None':
		    biker_bornIn.append(1952)
		else:
		    biker_bornIn.append(int(byear))

		dt = datetime.strptime(date, format)
		tour_day.append(dt.day)
		tour_month.append(dt.month)
		tour_year.append(dt.year)

		try:
		    age_while_tour.append(dt.year-int(byear))
		except:
		    age_while_tour.append(dt.year-1952)

		mdt = datetime.strptime(mdate, format)
		biker_member_since_day.append(mdt.day)
		biker_member_since_month.append(mdt.month)
		biker_member_since_year.append(mdt.year)
		
		diff = dt - mdt
		member_how_long_days.append(diff.days)
		
	
	bikers_network = pd.read_csv('./data/bikers_network.csv')
	bikers_network.index = bikers_network['biker_id']
	bikers_network.drop(['biker_id'], axis=1, inplace=True)
	tour_convoy = pd.read_csv('./data/tour_convoy.csv')
	tour_convoy.index = tour_convoy['tour_id']
	tour_convoy.drop(['tour_id'], axis=1, inplace=True)
	
	for index in dataset.index:
		tour_id = dataset.iloc[index]['tour_id']
		biker_id = dataset.iloc[index]['biker_id']
		host_id = dataset.iloc[index]['host_id']

		try:
		    friends = bikers_network.loc[biker_id]['friends'].split()
		    num_friends = len(friends)
		except:
		    friends = []
		    num_friends = 1
		try:
		    going = tour_convoy.loc[tour_id]['going'].split()
		except:
		    going = []
		try:
		    not_going = tour_convoy.loc[tour_id]['not_going'].split()
		except:
		    not_going = []
		try:
		    maybe = tour_convoy.loc[tour_id]['maybe'].split()
		except:
		    maybe = []
		try:
		    invited = tour_convoy.loc[tour_id]['invited'].split()
		except:
		    invited = []

		num_going, num_not_going, num_maybe, num_invited = 0, 0, 0, 0
		for friend in friends:
		    if friend in going:
		        num_going += 1
		    elif friend in not_going:
		        num_not_going += 1
		    elif friend in maybe:
		        num_maybe += 1
		    if friend in invited:
		        num_invited += 1
		    
		num_friends_going.append(num_going)
		num_friends_not_going.append(num_not_going)
		num_friends_maybe.append(num_maybe)
		num_friends_invited.append(num_invited)

		perc_friends_going.append(100*num_going/num_friends)
		perc_friends_not_going.append(100*num_not_going/num_friends)
		perc_friends_maybe.append(100*num_maybe/num_friends)
		perc_friends_invited.append(100*num_invited/num_friends)

		tour_going.append(len(going))
		tour_not_going.append(len(not_going))
		tour_maybe.append(len(maybe))

		is_host_friend.append(1 if host_id in friends else 0)
		
	dataset.drop(['timestamp', 'language_id', 'location_id', 'member_since', 
                  'tour_date', 'area', 'city', 'state', 'country', 'pincode'], axis=1, inplace=True)
	dataset['timestamp_year'] = timestamp_year
	dataset['timestamp_month'] = timestamp_month
	dataset['timestamp_day'] = timestamp_day
	dataset['timestamp_hour'] = timestamp_hour
	dataset['timestamp_min'] = timestamp_min
	dataset['timestamp_sec'] = timestamp_sec
	dataset['bornIn'] = biker_bornIn
	dataset['tour_day'] = tour_day
	dataset['tour_month'] = tour_month
	dataset['tour_year'] = tour_year
	dataset['tour_latitude'] = tour_latitude
	dataset['tour_longitude'] = tour_longitude
	dataset['member_since_day'] = biker_member_since_day
	dataset['member_since_month'] = biker_member_since_month
	dataset['member_since_year'] = biker_member_since_year
	dataset['member_how_long_days'] = member_how_long_days
	dataset['biker_time_zone'] = biker_time_zone
	dataset['time_zone_diff'] = abs(np.array(biker_time_zone) - np.array(tour_time_zone))
	dataset['biker_age_while_tour'] = age_while_tour
	dataset['num_friends_going'] = num_friends_going
	dataset['num_friends_not_going'] = num_friends_not_going
	dataset['num_friends_maybe'] = num_friends_maybe
	dataset['num_friends_invited'] = num_friends_invited
	dataset['perc_friends_going'] = perc_friends_going
	dataset['perc_friends_not_going'] = perc_friends_not_going
	dataset['perc_friends_maybe'] = perc_friends_maybe
	dataset['perc_friends_invited'] = perc_friends_invited
	dataset['tour_going'] = tour_going
	dataset['tour_not_going'] = tour_not_going
	dataset['tour_maybe'] = tour_maybe
	dataset['is_host_friend'] = is_host_friend
	dataset['tour_popularity'] = np.array(tour_going) + 0.5*np.array(tour_maybe)
	dataset = pd.get_dummies(dataset, columns=['gender'])
	
	if dset == 'train':
		like = list(dataset['like'])
		labels = pd.DataFrame(data=like, columns=['like'])
		labels['biker_id'] = dataset['biker_id']
		labels['tour_id'] = dataset['tour_id']
		dataset.drop(['like', 'dislike'], axis=1, inplace=True)
		labels.to_csv('./reqd data/y_{0}.csv'.format(dset), index=False)

	dataset.to_csv('./reqd data/X_{0}.csv'.format(dset), index=False)

def predict(params, weight=1, num=1):
	X_train = pd.read_csv('./reqd data/X_train.csv')
	X_train.drop(['host_id'], axis=1, inplace=True)
	y_train = pd.read_csv('./reqd data/y_train.csv')
	X_test = pd.read_csv('./reqd data/X_test.csv')
	X_test.drop(['host_id'], axis=1, inplace=True)

	all_ids = X_train['biker_id'].unique()
	num_ids = len(all_ids)

	kfold = KFold(n_splits=5)
	pred_prob = []

	for t_indx, v_indx in kfold.split(all_ids):
		
		X_train = pd.read_csv('./reqd data/X_train.csv')
		X_train.drop(['host_id'], axis=1, inplace=True)
		y_train = pd.read_csv('./reqd data/y_train.csv')
		X_test = pd.read_csv('./reqd data/X_test.csv')
		X_test.drop(['host_id'], axis=1, inplace=True)

		train_ids = all_ids[t_indx]
		val_ids = all_ids[v_indx]
		test_ids = X_test['biker_id'].unique()

		train_indx, val_indx = [], []
		for id in all_ids:
		    if id in train_ids:
		        train_indx.extend(np.where(X_train['biker_id'] == id)[0])
		    else:
		        val_indx.extend(np.where(X_train['biker_id'] == id)[0])

		train_bikers = X_train.iloc[train_indx]['biker_id']
		val_bikers = X_train.iloc[val_indx]['biker_id']
		test_bikers = X_test['biker_id']
		test_tours = X_test['tour_id']

		X_train = X_train.drop(['biker_id', 'tour_id'], axis=1).to_numpy()
		X_test = X_test.drop(['biker_id', 'tour_id'], axis=1).to_numpy()
		y_train = y_train['like'].to_numpy()

		X_val = X_train[val_indx, :]
		y_val = y_train[val_indx]
		X_train = X_train[train_indx, :]
		y_train = y_train[train_indx]

		X_train_1 = X_train[y_train==1]
		X_train_0 = X_train[y_train==0]
		X_train_norm = np.concatenate((X_train_0, np.tile(X_train_1, (weight, 1))), axis=0)
		y_train_norm = np.concatenate((np.zeros(X_train_0.shape[0]), np.ones(weight*X_train_1.shape[0])), axis=0)
		X_train_norm, y_train_norm = shuffle(X_train_norm, y_train_norm, random_state=0)

		sclr = StandardScaler()
		X_train_norm = sclr.fit_transform(X_train_norm)
		X_val = sclr.transform(X_val)
		X_test = sclr.transform(X_test)
		
		train_set = lgbm.Dataset(X_train_norm, y_train_norm)
		val_set = lgbm.Dataset(X_val, y_val)
		
		model = lgbm.train(params, train_set=train_set, valid_sets=val_set)
		
		y_train_pred = model.predict(X_train_norm)
		y_test_pred_prob = model.predict(X_test)
		pred_prob.append(y_test_pred_prob)
		
	y_test_pred_prob_avg = sum(pred_prob)/len(pred_prob)
		
	y_test_pred_ord = []
	biker_tours = []
	for id in test_ids:
		indx = np.where(test_bikers==id)[0]
		biker_tours.append(list(test_tours[indx]))
		pred = np.flip(np.argsort(y_test_pred_prob_avg[indx]))
		y_test_pred_ord.append(pred)

	pred_array = []
	for y_test_pred, tours in zip(y_test_pred_ord, biker_tours):
		string = ''
		for idx in y_test_pred:
		    string += tours[idx]
		    string += ' '
		pred_array.append(string)
		
	pred = pd.DataFrame(pred_array, columns=['tour_id'])
	pred['biker_id'] = list(test_ids)
	pred.to_csv('./ME17B122_ME17B156_{0}.csv'.format(num), index=False)
	
os.mkdir('./reqd data')
print('Creating training set...')
create_dataset(dset='train')
print('Creating test set...')
create_dataset(dset='test')

params1 = {'metric': 'f1', 'learning_rate': 0.05,
           'colsample_bytree': 0.6, 'subsample': 0.6,
           'num_leaves':32}

params2 = {'metric': 'f1', 'learning_rate': 0.05,
           'colsample_bytree': 0.5, 'subsample': 0.5,
           'num_leaves':32}
           
print('Calculating 1st prediction...')
predict(params1, weight=1, num=1)
print('Calculating 2nd prediction...')
predict(params2, weight=1, num=2)
print('Deleting created folders...')
os.remove('./reqd data/X_train.csv')
os.remove('./reqd data/y_train.csv')
os.remove('./reqd data/X_test.csv')
os.rmdir('./reqd data')
print('Done.')
