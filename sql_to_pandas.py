import os
import os.path

from pycds import Station
from pandas import read_sql_query
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


def ensure_path_exists(path):
    root_dir = os.path.dirname(path)
    os.makedirs(root_dir, exist_ok=True)


def pickle_station_from_db(stn_id, path):
    q = sesh.execute('SELECT query_one_station(:id)', {'id': stn_id});
    text, = q.fetchone()
    x = read_sql_query(text, engine)
    ensure_path_exists(path)
    x.to_pickle(path)


if __name__ == '__main__':

    engine = create_engine('postgresql://hiebert@db3/crmp')
    Session = sessionmaker(bind=engine)
    sesh = Session()

    stations = sesh.query(Station).all()

    for station in stations:
        net_name, native_id = station.network.name, station.native_id
        if len(station.histories) < 1 or len(station.network.variables) < 1:
            print("Skipping {}, {}".format(net_name, native_id))
            continue
        print("Processing {}, {}".format(net_name, native_id))
        path = os.path.join('station_dump', net_name,
                            native_id + '.pickle')
        try:
            pickle_station_from_db(station.id, path)
        except Exception as e:
            print("Error on {}, {}".format(net_name, native_id))

