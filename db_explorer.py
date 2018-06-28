import pyodbc
import numpy as np
import pandas as pd
import string
import feature_engineering as fe
import json
import config


''' still to do:
        - save column class scores in dictionary
        - save dictionary as JSON file
        - train classifier to predict other cases
        - include within-column variance as training feature
        - output summary of tables in database
'''

# Configure DB connection
conn = pyodbc.connect('DRIVER={SQL Server};'
                      'SERVER=' + config.server +
                      'DATABASE=' + config.database +
                      'UID=' + config.username +
                      'PWD=' + config.password)

# train decision tree
clf = fe.train_model()


def classify_rows(column_df, clf):
    ''' classify every row within a column, returning confidences of each
        column type label

    Args:
        column_df (pandas.DataFrame): column from a database
        clf (obj): trained classifer, must possess .predict() function

    Returns:
        row_scores (dict): label confidences for column
    '''

    # create features from raw data
    col_df = fe.create_features(column_df)

    num_rows = len(col_df)

    row_scores = {}
    for row in range(len(col_df)):
        # classify each row
        label = clf.predict(col_df.iloc[row,1:].values.reshape(1,-1))[0]

        # add label to col_scores
        if label in row_scores.keys():
            row_scores[label][0] += 1
            row_scores[label][1] = 100*row_scores[label][0]/num_rows
        else:
            row_scores[label] = [1, 1/num_rows]

    # store only the percentage score
    for key in row_scores.keys():
        value = row_scores[key]
        row_scores[key] = value[1]

    return(row_scores)


def query_to_dataframe(query, cursor):
    ''' convert SQL query to Pandas DataFrame

    Args:
        query (str): sql query to execute on database
        cursor (obj): cursor object linked to database

    Returns:
        df (DataFrame): result of sql query
    '''

    df = []  # where to store rows from query result

    # add each row from query to list
    for row in cursor.execute(query):
        df.append(row)

    # convert list to pandas data frame
    df = pd.DataFrame(np.vstack(row for row in df))

    # assign column names
    df.columns = pd.DataFrame(np.matrix(cursor.description))[0].values

    return df


def mask_database(connection):
    ''' find columns in all tables of a database which may contain sensitive
        information.

    Args:
        cursor (obj): cursor object linked to database

    Returns:
        json (str): json string of the following format:
        {
            table_name {
                column_name {
                    ExampleData: [
                        row12,
                        row42, ...
                        row87
                    ],
                    Scores: {
                        first_name: ...,
                        area: ...
                    },
                    columnMeta {
                        type: ...,
                        keys: ...
                    }
                },
                tableMeta {
                    num_cols: ...,
                    num_rows: ...,
                    pk_name: ...,
                    pkcolumn_name: ...
                }
            }
        }
    '''

    # create cursor object
    cursor = connection.cursor()

    # create dictionary in which to store metadata and column labels
    d = {}

    # find all tables in database, save as list of table names
    tables = []
    for table in cursor.tables():
        if table.table_type == 'TABLE' and table.table_name[:8] != 'trace_xe':
            table_name = table.table_name
            tables.append(table.table_name)

    # classify and describe all tables
    for table in tables:
        d[table] = {}
        d[table]['tableMeta'] = {}
        meta = d[table]['tableMeta']

        # primary key information
        for key in cursor.primaryKeys(table):
            meta['pkcolumn_name'] = key.column_name
            meta['pk_name'] = key.pk_name

        # foreign key information
        for key in cursor.foreignKeys(table):
            meta['fk_name'] = key.fk_name

        # number of rows
        num_rows = 'SELECT COUNT(*) FROM ' + table
        cursor.execute(num_rows)
        num_rows = cursor.fetchone()
        meta['num_rows'] = num_rows[0]

        # convert table to pandas data frame
        query = 'SELECT TOP 1000 * FROM ' + table
        df = query_to_dataframe(query, cursor)
        meta['num_cols'] = df.shape[1]

        # classify each column in table and store metadata
        for col in df:
            d[table][col] = {'Scores': 0,
                             'ColumnMeta': {},
                             'ExampleData': 0}

            col_df = pd.DataFrame(df[col].dropna())
            # label raw data column
            col_df.columns = ['Data']

            # sample 5 random rows
            sample_5 = (col_df['Data'].sample(n=5, axis=0).tolist())
            d[table][col]['ExampleData'] = sample_5

            # classify rows in dataframe
            scores = classify_rows(col_df, clf)
            d[table][col]['Scores'] = scores

        # column metadata
        cursor.execute('EXEC sp_help \'' + table + '\'')
        cursor.nextset()
        table_info = cursor.fetchall()
        for row in table_info:
            col = row[0]
            col_meta = d[table][col]['ColumnMeta']
            col_meta['Type'] = row[1]
            col_meta['Computed'] = row[2]
            col_meta['Length'] = row[3]
            col_meta['Nullable'] = row[6]
            col_meta['TrimTrailingBlanks'] = row[7]
            col_meta['FixedLenNullInSource'] = row[8]
            col_meta['Collation'] = row[9]

    # save result as json file
    json_string = json.dumps(d, indent=4, sort_keys=True)
    outfile = open('output.json', 'w')
    outfile.write(json_string)
    outfile.close()
    cursor.close()


mask_database(conn)
conn.close()
