#!/usr/bin/python
# -*- coding: utf-8 -*-

import sqlalchemy as sa
import re
import MySQLdb, os, sys, shutil

#MySql DB functions:
def commit_db(db,sql,args=None):
    try:
        # Execute the SQL command
        db.cursor().execute(sql,args)
        # Commit your changes in the database
        db.commit()
        return True
    except MySQLdb.Error, e:
        #print("Something went wrong: {}".format(e))
        #print 'trying hack:'
        try:
            tmpargs=list(args)
            tmpargs[4]=re.sub('[\W_]+', ' ', tmpargs[4])
            tmpargs[7]=re.sub('[\W_]+', ' ', tmpargs[7])
            #change this to: ^[a-zæøåA-ZÆØÅ0-9_]+( [a-zæøåA-ZÆØÅ0-9_]+)*$ and include æøå http://stackoverflow.com/questions/15472764/regular-expression-to-allow-spaces-between-words
            args=tuple(tmpargs)
            # Execute the SQL command
            db.cursor().execute(sql, args)
            # Commit your changes in the database
            db.commit()
            return True
        except MySQLdb.Error, e:
            print("Something went wrong: {}".format(e))
            # Rollback in case there is any error
            db.rollback()
            return False

def connect_to_database():
    """ Connect to the database 'mf_db'.
	Return:
	cur -- a MySQLdb cursor
	con -- a connection to the database
	"""
    con = MySQLdb.connect('drmedieforsk01', 'mfdev', 'admin', 'spredfast', charset='utf8')
    con.set_character_set('utf8')
    return con


# return MySQLdb.connect('localhost', 'root', 'admin', 'medieforskning','charset='utf8')

def check_if_table_exist(cursor, table):
    """ Check if a specified table exist in the database. """
    sql = "SHOW TABLES LIKE '" + table + "'"
    cursor.execute(sql)
    result = cursor.fetchone()
    if result: return True
    else: return False

def create_if_table_not_exists(db, table, fields):
    """ Check if a specified table exist in the database. """
    #if not check_if_table_exist(db.cursor(), table):
    sql = "CREATE TABLE if not exists " + table + fields
    commit_db(db,sql)

def delete_table(cursor, table):
    """ Delete the specified table from the database. """
    if check_if_table_exist(cursor, table):
        sql = "DROP TABLE " + table
        cursor.execute(sql)

def create_table_index(cursor, table):
    """ Delete the specified table from the database. """
    if check_if_table_exist(cursor, table):
        sql = "CREATE UNIQUE INDEX 'unique_key' ON " + table + " (unique_key)"
        cursor.execute(sql)