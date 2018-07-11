import pandas as pd
import numpy as np
from random import sample
import string
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle as pkl
# from tpot import TPOTClassifier

# load raw reference data
names_set = set(pd.read_csv('./ref/names.csv')['Name'].values)
names = [name for name in names_set]
surnames_set = set(pd.read_csv('./ref/surnames.csv')['Surname'].values)
surnames = [surname for surname in surnames_set]
companies = [company for company in
             set(pd.read_csv('./ref/companies.csv')['Company'].values)]
phones = [number for number in
          pd.read_csv('./ref/phones.csv')['phone'].values]
dates = [date for date in
         pd.read_csv('./ref/birthdays.csv')['DoB'].values]
# create dummy gender data
genders = []
for i in range(1291):
    genders.append(sample(['male', 'female'], 1))
drivers = [license for license in
          pd.read_csv('./ref/drivers-licenses.csv')['DriversLicense'].values]

# data relating to address
addresses = pd.read_csv('./ref/nz-street-address.csv')
street_nos = addresses['StreetNumber'].values
cities = addresses['City'].values
cities_set = set(addresses['City'].values)
areas = addresses['Area'].values
areas_set = set(addresses['City'].values)
postcodes = addresses['Postcode'].values
streets = addresses['StreetName'].values
full_addresses = addresses['FullAddress'].values

# append labels to each of the training observations
names = np.vstack([name, 'first_name'] for name in sample(names,1291))
surnames = np.vstack([surname, 'last_name'] for surname in sample(surnames,1291))
full_names = np.core.defchararray.add(np.core.defchararray.add(names[:,0], ' '),
                                      surnames[:,0])
full_names = np.vstack([name, 'full_name'] for name in full_names)
genders = np.vstack([gender, 'gender'] for gender in genders)
street_nos = np.vstack([number, 'street_no'] for number in street_nos)
streets = np.vstack([street, 'street'] for street in streets)
cities = np.vstack([city, 'city'] for city in cities)
areas = np.vstack([area, 'area'] for area in areas)
postcodes = np.vstack([code, 'postcode'] for code in postcodes)
full_addresses = np.vstack([address, 'full_address'] for address in full_addresses)
companies = np.vstack([company, 'employer'] for company in sample(companies,1291))
phones = np.vstack([phone, 'phone'] for phone in phones)
dates = np.vstack([date, 'date'] for date in dates)
drivers = np.vstack([license, 'drivers_license'] for license in drivers)

# stack data-label pairs together to form raw data frame
raw_df = np.vstack(col for col in [names,
                                   surnames,
                                   full_names,
                                   genders,
                                   street_nos,
                                   streets,
                                   cities,
                                   areas,
                                   postcodes,
                                   full_addresses,
                                   companies,
                                   phones,
                                   dates,
                                   drivers])

raw_df = pd.DataFrame(raw_df)
raw_df.columns = ['Data', 'Label']

# road suffixes
road_suffixes = pd.read_csv('./ref/road_abbrv.csv')['RoadSuffix'].values

# company suffixes
company_suffixes = {'Inc',
                    'Co',
                    'Corp',
                    'Ltd',
                    'Pty',
                    'LLC',
                    'Limited',
                    'Group'}


def remove_punctuation(words):
    ''' remove all punctuation except for dashes (found in dates) '''

    punc = set(string.punctuation)
    punc.remove('-')
    return ''.join([char for char in words if char not in punc])


def count_digits(row_val):
    ''' count number of digits in a given table cell '''

    # convert to string
    row_val = str(row_val)

    # count digits
    num_digits = sum(c.isdigit() for c in row_val)

    return num_digits


def count_letters(row_val):
    ''' count number of letters in a given table cell '''

    # convert to string
    row_val = str(row_val)

    # count letters
    num_letters = sum(c.isalpha() for c in row_val)

    return num_letters


def count_spaces(row_val):
    ''' count number of spaces in a given table cell '''

    # convert to string
    row_val = str(row_val)

    # count digits
    num_spaces = sum(c.isspace() for c in row_val)

    return num_spaces


def create_features(df):
    ''' perform feature engineering on a representative personal information
    data set as preparation for training.

    Args:
        df (DataFrame): [raw_data, label]

    Returns:
        df_out (DataFrame): [Data, num_words, num_chars, ..., label]
    '''

    # remove punctuation
    df['Data'] = df['Data'].apply(remove_punctuation)

    # remove trailing whitespace
    df['Data'] = df['Data'].apply(lambda x : x.strip())

    # is first char a letter?
    df['first_letter'] = df['Data'].apply(lambda x: str(x)[0] in
                                          string.ascii_letters).astype(int)

    # how many words?
    df['num_words'] = df['Data'].apply(lambda x: len(str(x).split(' ')))

    # how many chars?
    df['num_chars'] = df['Data'].apply(lambda x: len(str(x)))

    # how many letters?
    df['num_letters'] =  df['Data'].apply(lambda x: count_letters(x))

    # how many digits?
    df['num_digits'] = df['Data'].apply(lambda x: count_digits(x))

    # how many dashes?
    df['num_dashes'] = df['Data'].apply(lambda x:
                                        str(x).count('-') + str(x).count('/'))

    # how many spaces?
    df['num_spaces'] =  df['Data'].apply(lambda x: count_spaces(x))

    # starts with first name?
    df['begins_name'] = df['Data'].apply(lambda x: str(x).split(' ')[0] in
                                      names_set).astype(int)

    # ends with surname?
    df['ends_surname'] = df['Data'].apply(lambda x: str(x).split(' ')[-1] in
                                          surnames_set).astype(int)

    # contains gender descriptor?
    df['has_gender'] = df['Data'].apply(lambda x:
                                        str(x).lower() in ['male',
                                                           'female',
                                                           'm',
                                                           'f']).astype(int)

    # contains road suffix?
    df['has_road'] = df['Data'].apply(lambda x:
                                      any(i in road_suffixes for i in
                                      str(x).split(' '))).astype(int)

    # contains city name?
    df['has_city'] = df['Data'].apply(lambda x:
                                      any(i in str(x) for i in
                                      cities_set)).astype(int)

    # contains area?
    df['has_area'] = df['Data'].apply(lambda x:
                                      any(i in str(x) for i in
                                      areas_set)).astype(int)

    # contains company name?
    df['has_company'] = df['Data'].apply(lambda x:
                                         any(i in company_suffixes for i in
                                         str(x).split(' '))).astype(int)

    return(df)


def train_model():
    # process raw data into training data frame
    train_df = create_features(raw_df)
    X_train = train_df.iloc[:,2:]
    y_train = train_df.iloc[:,1]

    pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        DecisionTreeClassifier(criterion='entropy', max_depth=10,
                               min_samples_leaf=2, min_samples_split=17)
    )

    pipeline.fit(X_train, y_train)

    # save the model to disk
    filename = './model/model.sav'
    pkl.dump(pipeline, open(filename, 'wb'))

    return(pipeline)


def optimize_pipeline():

    train_df = create_features(raw_df)
    X_train = train_df.iloc[:,2:]
    y_train = train_df.iloc[:,1]

    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                        random_state=42, verbosity=2)

    pipeline_optimizer.fit(X_train, y_train)

    pipeline_optimizer.export('tpot_exported_pipeline.py')


train_model()
