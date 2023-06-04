""" 
Module with functions to extract data from NASDAQ API and given csv file,
transform extracted data and load it to PostgreSQL database. To do so, 
functions use PySpark and psycopg2 libraries.
"""
import os
import configparser
import sys
from dotenv import load_dotenv
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, DoubleType
from pyspark.sql.functions import split, col, quarter, udf
import pandas as pd
from pyspark.sql import SparkSession
from typing import List, Tuple
import requests
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pyspark

def get_path_to_spark_jars()->str:
    """ Function to get path to Spark jars from .env file.

    Returns:
        str: path to Spark jars as a string
    """
    load_dotenv()
    path = os.getenv('JDBC_DRIVER_PATH')
    if not path:
        raise ValueError("JDBC_DRIVER_PATH is not available")
    return path

# set PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON to sys.executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# create SparkSession with jdbc driver
spark = SparkSession.builder.appName('BigMacData').config("spark.jars", get_path_to_spark_jars()).getOrCreate()

# set heartbeat interval to 3600 milliseconds

conf = pyspark.SparkConf()
conf.set("spark.executor.heartbeatInterval", "3600s")

def get_api_key()->str:
    """ Function to get NASDAQ API key from .env file.

    Returns:
        str: NASDAQ API key as a string
    """
    load_dotenv()
    key = os.getenv('NASDAQ_KEY')
    if not key:
        raise ValueError("NASDAQ_KEY \
                         is not available")
    return key

# function to get data from NASDAQ API for chosen country from start_date to end_date
def get_data(country_code: str, start_date: str, end_date: str)->List[List[str]]:
    """ Function to get data from NASDAQ API for chosen country from chosen period of time (start_date to end_date).

    Args:
        country_code (str): code used by NASDAQ API to identify country, e.g. 'POL' for Poland
        start_date (str): starting date which is used in API request, format YYYY-MM-DD
        end_date (str): ending date which is used in API request, format YYYY-MM-DD

    Returns:
        List[List[str]]: List of lists with data from NASDAQ API, each list contains data for one day in format:
        [country_code, date, local_price, dollar_price, dollar_ppp] where:
            * country_code - code used by NASDAQ API to identify country, e.g. 'POL' for Poland
            * date - date in format YYYY-MM-DD
            * local_price - price of BigMac in local currency
            * dollar_price - price of BigMac in USD
            * dollar_ppp - price of BigMac in USD adjusted by GDP per capita
    """
    api_k = get_api_key()

    url = f"https://data.nasdaq.com/api/v3/datasets/ECONOMIST/BIGMAC_{country_code}?start_date={start_date}&end_date={end_date}&api_key={api_k}"

    r = requests.get(url)

    # check if data is available, if not return ValueError
    if r.status_code == 200:
        data = r.json()
        # we are interested only in columns in ['Date', 'local_price', 'dollar_price', 'dollar_ppp']
        # check index of these columns
        columns = data['dataset']['column_names']
        columns = [columns.index(column) for column in columns if column in ['Date',
                                                                             'local_price',
                                                                             'dollar_price',
                                                                             'dollar_ppp']]
        # get data from json
        data = data['dataset']['data']
        # get only columns we are interested in
        data = [[row[i] for i in columns] for row in data]
        # add country_code to data
        data = [[country_code] + row for row in data]

        # add index_id to data which will be used as a primary key
        # index_id is a combination of country_code and date
        data = [[row[0] + '_' + row[1].replace('-', '')] + row for row in data]

        return data
    else:
        raise ValueError(f"Data for country {country_code} is not available")

def get_data_df(countries: List[str], start_date: str,
                end_date: str, saving: bool = True)->pyspark.sql.dataframe.DataFrame:
    """ Function to get data from NASDAQ API for chosen countries from chosen period of time (start_date to end_date).
        Function returns DataFrame with data from NASDAQ API and can save it to csv file.

    Args:
        countries (List[str]): list of country codes used by NASDAQ API to identify countries, e.g. ['POL', 'USA']
        start_date (str): start date in format YYYY-MM-DD
        end_date (str): end date in format YYYY-MM-DD
        saving (bool, optional): Boolean value which indicates if DataFrame should be saved to csv file. Defaults to True.

    Returns:
        pyspark.sql.dataframe.DataFrame: PySpark DataFrame with data from NASDAQ API 
                                         for chosen countries from chosen period of time 
                                         (start_date to end_date).
    """

    # Destination file name is 'BigMacData_{start_date}_{end_date}.csv'
    file_name = f'BigMacData_{start_date}_{end_date}.csv'

    # create an empty list
    data = []

    # iterate over countries list and get data from NASDAQ API
    for country_code in countries:
        try:
            data += get_data(country_code, start_date, end_date)
        except ValueError:
            print(f"Data for country {country_code} is not available")
        except requests.exceptions.ConnectionError:
            print(f"Connection error for country {country_code}")
        except requests.exceptions.Timeout:
            print(f"Timeout for country {country_code}")

    schema = StructType([StructField('INDEX_ID', StringType(), True),
                         StructField('COUNTRY_CODE', StringType(), True),
                         StructField('DATE', DateType(), True),
                         StructField('LOCAL_PRICE', DoubleType(), True),
                         StructField('DOLLAR_PRICE', DoubleType(), True),
                         StructField('DOLLAR_PPP', DoubleType(), True)])

    # create DataFrame from data
    df = spark.createDataFrame(data, schema = ['INDEX_ID',
                                               'COUNTRY_CODE',
                                               'DATE',
                                               'LOCAL_PRICE',
                                               'DOLLAR_PRICE',
                                               'DOLLAR_PPP'])
    # transform DATE column to datetime format
    df = df.withColumn('DATE', df['DATE'].cast(DateType()))
    df = spark.createDataFrame(df.rdd, schema)

    # save DataFrame to csv file in the same directory as the script
    if saving:
        df2 = df.toPandas()
        path_csv = os.path.join(os.path.dirname(__file__), file_name)
        df2.to_csv(path_csv, index=False)

    return df

def read_config()->Tuple[str, str]:
    """ Function to read start_date and end_date from config.ini file

    Returns:
        Tuple[str, str]: start_date and end_date from config.ini file
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

    start_date = config['NASDAQ']['START_DATE']
    end_date = config['NASDAQ']['END_DATE']

    return (start_date, end_date)

@udf(returnType=IntegerType())
def get_financial_year(date)->int:
    """ Function to get financial year from date, 
        financial year starts on 1st of July and ends on 30th of June.

    Args:
        date (str): date in format YYYY-MM-DD 

    Returns:
        int: financial year from given date as an integer
    """
    # change date to datetime format
    date = pd.to_datetime(date)

    if date.month < 7:
        return date.year - 1
    else:
        return date.year

def get_db_credentials()->Tuple[str, int, str, str, str]:
    """ Function to get database credentials from .env file.
        Those credentials are used to connect to PostgreSQL database.

    Returns:
        Tuple[str, str, str, str, str]: host, port, db_name, user, password for PostgreSQL database
                                        read from .env file.
    """
    load_dotenv()
    host = os.getenv('POSTGRES_SERVER')
    port = os.getenv('POSTGRES_PORT')
    db_name = os.getenv('POSTGRES_DBNAME')
    user = os.getenv('POSTGRES_USERNAME')
    password = os.getenv('POSTGRES_PASSWORD')

    # if any of the credentials is not available raise ValueError
    if not host:
        raise ValueError("POSTGRES_SERVER is not available")
    if not port:
        raise ValueError("POSTGRES_PORT is not available")
    if not db_name:
        raise ValueError("POSTGRES_DBNAME is not available")
    if not user:
        raise ValueError("POSTGRES_USERNAME is not available")
    if not password:
        raise ValueError("POSTGRES_PASSWORD is not available")

    return (host, int(port), db_name, user, password)

# create database
def create_db(host: str, port: int, db_name: str, user: str, password: str)->None:
    """ Function to create PostgreSQL database. It connects to PostgreSQL database 
        by psycopg2 library and creates database with given name. Initially it connects
        to default database 'postgres' and then creates new database.

    Args:
        host (str): host name or IP address of PostgreSQL database
        port (str): port number of PostgreSQL database
        db_name (str): database name to be created
        user (str): _description_
        password (str): _description_
    """
    # create connection to PostgreSQL database
    conn = psycopg2.connect(host=host,
                            port=port,
                            dbname="postgres",
                            user=user,
                            password=password,
                            options='-c statement_timeout=10s')

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    print("> Connected to PostgreSQL database")
    # check if database with given name already exists
    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
    exists = cur.fetchone()
    # if database with given name does not exist create new database
    if not exists:
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"> Database {db_name} created")
    else:
        print(f"> Database {db_name} already exists")
    # close connection to PostgreSQL database
    cur.close()
    conn.close()

def create_db_table(host: str, port: int, db_name: str, user: str, password: str)->None:
    """ Function to create tables in PostgreSQL database.
    
    Args:
        host (str): host for PostgreSQL database
        port (int): port for PostgreSQL database
        db_name (str): name of the database
        user (str): username for PostgreSQL database
        password (str): password for PostgreSQL database
        
        Returns: None        
    """
    conn = psycopg2.connect(host=host,
                            port=port,
                            dbname=db_name,
                            user=user,
                            password=password,
                            options='-c statement_timeout=10s')
    cur = conn.cursor()

    # create table in PostgreSQL database
    cur.execute("""CREATE TABLE IF NOT EXISTS geography (
                    country_name VARCHAR(50) NOT NULL,
                    country_code VARCHAR(3),
                    region VARCHAR(50) NOT NULL,
                    date_start TIMESTAMP DEFAULT NOW(),
                    date_end TIMESTAMP DEFAULT NULL,
					PRIMARY KEY (country_code,date_start));""")

    cur.execute("""CREATE TABLE IF NOT EXISTS time (
                    index_id VARCHAR(15) PRIMARY KEY,
                    date DATE NOT NULL,
                    quarter INT NOT NULL,
                    financial_year INT NOT NULL
                    );""")

    cur.execute("""CREATE TABLE IF NOT EXISTS big_mac_index (
                    index_id VARCHAR(15) REFERENCES time(index_id),
                    country_code VARCHAR(3) REFERENCES geography(country_code),
                    local_price NUMERIC(15, 4) NOT NULL,
                    dollar_price NUMERIC(15, 4) NOT NULL,
                    dollar_ppp NUMERIC(15, 2) NOT NULL
                    );""")

    # create function to insert data into geography table
    cur.execute("""CREATE OR REPLACE function public.scd2()
                   returns trigger 
                   AS $BODY$
                       BEGIN
                           update geography 
                           set date_end = now()
                           where country_code=new.country_code
                           and date_end is null;
                           return new;
                      END;
                  $BODY$
                  LANGUAGE plpgsql;""")

    # create trigger to insert data into geography table
    cur.execute("""CREATE TRIGGER example_trigger before INSERT ON geography
                   FOR EACH ROW EXECUTE PROCEDURE scd2();""")

    # commit changes to database
    conn.commit()
    # close connection to PostgreSQL database
    cur.close()
    conn.close()

# function to load data from pyspark dataframe to PostgreSQL database
def load_data_to_db(host: str, port: int, db_name: str, user: str, password: str, df: pyspark.sql.DataFrame, table_name: str)->None:
    """ Function to load data from pyspark dataframe to PostgreSQL database.

    Args:
        host (str): host for PostgreSQL database
        port (int): port for PostgreSQL database
        db_name (str): database name for PostgreSQL database
        user (str): username for PostgreSQL database
        password (str): password for PostgreSQL database
        df (pyspark.sql.DataFrame): pyspark dataframe to be loaded to PostgreSQL database
        table_name (str): name of the table in PostgreSQL database to which data will be loaded

        Returns: None
    """
    try:
        # create connection to PostgreSQL database
        conn = psycopg2.connect(host=host,
                                port=port,
                                dbname=db_name,
                                user=user,
                                password=password,
                                options='-c statement_timeout=10s')
        # create cursor
        cur = conn.cursor()

        # create properties for connection to PostgreSQL database
        properties = {"user": user,
                      "password": password,
                      "driver": "org.postgresql.Driver"}
        
        # write data from pyspark dataframe to PostgreSQL database
        df.write.jdbc(url=f"jdbc:postgresql://{host}:{port}/{db_name}",
                      table=table_name,
                      mode='append',
                      properties=properties)

        # commit changes to database
        conn.commit()

        print(f"> Data was loaded to {table_name} table")

        cur.close()
    
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)
    
    finally:
        # close connection to PostgreSQL database
        if conn is not None:
            conn.close()

def main()->None:
    """ Main function which is responsible for loading data from csv file and NASDAQ API, 
        transforming them and saving it to PostgreSQL database.
    """

    # EXTRACT DATA FROM CSV FILES
    # File is stored in the same directory as the script
    path_csv = os.path.join(os.path.dirname(__file__), 'economist_country_codes.csv')
    geography_df = spark.read.csv(path_csv, 
                                  header=True, 
                                  inferSchema=True, 
                                  sep=';')

    print("> Geography DataFrame was loaded from csv file")
    # TRANSFORM DATAFRAME
    # Split column 'COUNTRY|CODE' into two columns: 'COUNTRY_NAME' and 'COUNTRY_CODE'
    geography_df = geography_df.withColumn('COUNTRY_NAME', split(col('COUNTRY|CODE'), '\|')[0])
    geography_df = geography_df.withColumn('COUNTRY_CODE', split(col('COUNTRY|CODE'), '\|')[1])
    # Drop column 'COUNTRY|CODE'
    geography_df = geography_df.drop('COUNTRY|CODE')
    # Change column order
    geography_df = geography_df.select('COUNTRY_NAME', 'COUNTRY_CODE', 'REGION')

    print("> Geography DataFrame was transformed")

    # get COUNTRY_CODE from dataframe and convert to list
    countries_codes = geography_df.select('COUNTRY_CODE').collect()
    countries_codes = [row.COUNTRY_CODE for row in countries_codes]

    # EXTRACT DATA FROM NASDAQ API
    (start_date, end_date) = read_config()

    # get data from NASDAQ API and save to csv file
    nasdaq_df = get_data_df(countries_codes, start_date, end_date)
    print("> Data was extracted from NASDAQ API")

    time_df = nasdaq_df.select('INDEX_ID', 'DATE')
    time_df = time_df.withColumn('QUARTER', quarter('DATE'))
    time_df = time_df.withColumn('FINANCIAL_YEAR', get_financial_year('DATE'))
    print("> Time DataFrame was created")

    big_mac_df = nasdaq_df.select('INDEX_ID','COUNTRY_CODE',
                                  'LOCAL_PRICE', 'DOLLAR_PRICE', 
                                  'DOLLAR_PPP')
    print("> Big Mac DataFrame was created")

    # CREATE DATABASE
    (host, port, db_name, user, password) = get_db_credentials()
    create_db(host, port, db_name, user, password)

    # CREATE TABLES IN DATABASE
    create_db_table(host, port, db_name, user, password)

    # INSERT DATA TO DATABASE TABLES
    load_data_to_db(host, port, db_name, user, password, geography_df, 'geography')
    load_data_to_db(host, port, db_name, user, password, time_df, 'time')
    load_data_to_db(host, port, db_name, user, password, big_mac_df, 'big_mac_index')

    print("> Data was loaded to PostgreSQL database")
    print("**********\tETL process finished\t**********")

if __name__ == '__main__':
    main()
