# Database Masking
Use this application to find out which columns in a SQL database are likely to contain sensitive personal information.

## Instructions:

### 1. Train model
**Using Docker**:   
`$ docker run -v $(pwd)/model:/usr/src/app/model -d jimbabwe/column-classifier:latest`   

OR

**Running locally**:   
`$ python feature_engineering.py`

In your current working directory, you should now find a new folder named `model` with a file `model.sav` inside.

### 2. Run `db_explorer`
   2.1. Make sure you have an active database connection, then create a new file named `config.py` (use `dummy_config.py` as a template).   
   2.2. `$ python db_explorer.py`

You should now have a file named `output.json` in your current working directory.
