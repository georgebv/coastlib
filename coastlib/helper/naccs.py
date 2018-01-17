import h5py
import pandas as pd


path = r'D:\Work folders\desktop projects\4 Pelham Shoreline Stabilization' \
       r'\1 Data\STWAVE\Base conditions\NACCS_XH_SimB_Post0_SP0067_STWAVE05_Timeseries.h5'

def stwave(path):
    # Load storm h5 file
    h_file = h5py.File(path, 'r')
    storms = list(h_file.keys())

    # Parse each storm data
    mwd, mwp, pp, we, wd, wm, zmwh, dates, depths, sterics, id = [], [], [], [], [], [], [], [], [], [], []
    for storm in storms:
        mwd.extend(h_file[storm]['Mean Wave Direction'])
        mwp.extend(h_file[storm]['Mean Wave Period'])
        pp.extend(h_file[storm]['Peak Period'])
        we.extend(h_file[storm]['Water Elevation'])
        wd.extend(h_file[storm]['Wind Direction'])
        wm.extend(h_file[storm]['Wind Magnitude'])
        zmwh.extend(h_file[storm]['Zero Moment Wave Height'])
        dates.extend(h_file[storm]['yyyymmddHHMM'])
        id.extend([int(storm.split(' - ')[-1])] * len(h_file[storm]['Peak Period']))
        depths.extend([float(h_file[storm].attrs['Save Point Depth'].decode())] * len(h_file[storm]['Peak Period']))
        sterics.extend([float(h_file[storm].attrs['Steric Level'].decode())] * len(h_file[storm]['Peak Period']))

    # Parse datetime
    dtind = []
    for date in dates:
        value = str(date)
        dtind.append(
            pd.datetime(
                year=int(value[:4]),
                month=int(value[4:6]),
                day=int(value[6:8]),
                hour=int(value[8:10]),
                minute=int(value[10:12])
            )
        )

    # Prepare a DataFrame
    df = pd.DataFrame(data=mwd, index=dtind, columns=['Mean Wave Direction (deg)'])
    df['Mean Wave Period (s)'] = mwp
    df['Peak Period (s)'] = pp
    df['Water Elevation (m MSL)'] = we
    df['Wind Direction (deg)'] = wd
    df['Wind Magnitude (m/s)'] = wm
    df['Zero Moment Wave Height (m)'] = zmwh
    df['Storm id'] = id
    df['Save Point Depth (m MSL)'] = depths
    df['Steric Level (m MSL)'] = sterics

    return df
