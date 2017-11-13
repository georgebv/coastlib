import sys
import os.path
import osgeo.osr as osr

try:
    from osgeo import ogr
except ImportError:
    import ogr

# Argument processing
if len(sys.argv) != 2:
    print('')
    print('Usage: "s57.py enc_chart.000"')
    sys.exit(1)

s57filename = sys.argv[1]

# Specify the target cCSV file in the same directory
filename, file_extension = os.path.splitext(s57filename)
csvfilename = filename
csvfilename += '.csv'

# Open the S57 file, and find the SOUNDG layer
ds = ogr.Open(s57filename)

try:
    src_soundg = ds.GetLayerByName('SOUNDG')
except AttributeError:
    print('')
    print('Error: SOUNDG layer or source S-57 file not found.')
    sys.exit(1)

# Create the output CSV file
csv_driver = ogr.GetDriverByName('CSV')
csv_ds = csv_driver.CreateDataSource(csvfilename)

# Import CSR WGS84 (EPSG 4326)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

# Specify the new SOUNDG layer
csv_layer = csv_ds.CreateLayer('csv_soundg', srs, geom_type=ogr.wkbPoint25D)

src_defn = src_soundg.GetLayerDefn()
field_count = src_defn.GetFieldCount()

# Display progress message
print('')
print(csvfilename, "created ...")

# Copy the SOUNDG schema, duplicating the current fields, and create the LAT,
# LONG and DEPTH fields for the extracted Point and calculated data

out_mapping = []
for fld_index in range(field_count):

    src_fd = src_defn.GetFieldDefn(fld_index)
    fd = ogr.FieldDefn(src_fd.GetName(), src_fd.GetType())
    fd.SetWidth(src_fd.GetWidth())
    fd.SetPrecision(src_fd.GetPrecision())
    if csv_layer.CreateField(fd) != 0:
        out_mapping.append(-1)
    else:
        out_mapping.append(csv_layer.GetLayerDefn().GetFieldCount() - 1)

fd = ogr.FieldDefn('LAT', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATD', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATS', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATH', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATDMS', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONG', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGD', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGS', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGH', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGDMS', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('DEPTHM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(0)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('DEPTHF', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(0)
csv_layer.CreateField(fd)

# Display progress message
print("SOUNDG layer created ...")

#  Process the data for the new fields
feat = src_soundg.GetNextFeature()

# The degree symbol needs to be defined for use
deg = u"\u00b0"

# GetFeatureCount can be expensive, so ...
iCount = 0

while feat is not None:

    multi_geom = feat.GetGeometryRef()

    for iPnt in range(multi_geom.GetGeometryCount()):
        # Get the multipart data
        pnt = multi_geom.GetGeometryRef(iPnt)

        # Get the layer definition from the KML file
        feat1 = ogr.Feature(feature_def=csv_layer.GetLayerDefn())

        for fld_index in range(field_count):
            feat1.SetField(out_mapping[fld_index], feat.GetField(fld_index))

        # Get the LATITUDE in Degrees.Decimal format
        soundinglat = pnt.GetX(0)

        # Extract the Degrees
        latdegrees = int(abs(soundinglat))
        # Extract the decimals
        trunclatdegrees = (abs(soundinglat) - latdegrees)
        # Calculate the Minutes
        latminutes = int(trunclatdegrees * 60)
        # Calculate the Seconds
        latseconds = round(((trunclatdegrees * 3600) % 60), 2)

        # Negative LATITUDE is West of the Zero Meridian
        if soundinglat <= 0:
            lathemisphere = "W"
        else:
            lathemisphere = "E"

        # Assemble the LATDMS string
        latstring = str(latdegrees)
        # Encode the defined degree symbol as utf-8 and add it
        latstring += str(deg.encode('utf8'))
        latstring += " "
        latstring += str(latminutes)
        latstring += "' "
        latstring += str(latseconds)
        latstring += '" '
        latstring += lathemisphere

        # Write the LATITUDE fields
        feat1.SetField('LAT', soundinglat)
        feat1.SetField('LATD', latdegrees)
        feat1.SetField('LATM', latminutes)
        feat1.SetField('LATS', latseconds)
        feat1.SetField('LATH', lathemisphere)
        feat1.SetField('LATDMS', latstring)

        # Get the LONGITUDE in Degrees.Decimal format
        soundinglong = pnt.GetY(0)

        # Extract the Degrees
        longdegrees = int(abs(soundinglong))
        # Extract the decimals
        trunclongdegrees = (abs(soundinglong) - longdegrees)
        # Calculate the Minutes
        longminutes = int(trunclongdegrees * 60)
        # Calculate the Seconds
        longseconds = round(((trunclongdegrees * 3600) % 60), 2)

        # Negative LONGITUDE is South of the equator
        if soundinglong <= 0:
            longhemisphere = "S"
        else:
            longhemisphere = "N"

        # Assemble the LONGDMS string
        longstring = str(longdegrees)
        # Encode the defined degree symbol as utf-8 and add it
        longstring += str(deg.encode('utf8'))
        longstring += " "
        longstring += str(longminutes)
        longstring += "' "
        longstring += str(longseconds)
        longstring += '" '
        longstring += longhemisphere

        # Write the LONGITUDE fields
        feat1.SetField('LONG', soundinglong)
        feat1.SetField('LONGD', longdegrees)
        feat1.SetField('LONGM', longminutes)
        feat1.SetField('LONGS', longseconds)
        feat1.SetField('LONGH', longhemisphere)
        feat1.SetField('LONGDMS', longstring)

        # Use metres as provided and calculate feet
        soundingmetres = pnt.GetZ(0)
        soundingfeet = soundingmetres * 3.28084

        # Write the DEPTH fields
        feat1.SetField('DEPTHM', soundingmetres)
        feat1.SetField('DEPTHF', soundingfeet)

        # Create the feature
        feat1.SetGeometry(pnt)
        csv_layer.CreateFeature(feat1)
        feat1.Destroy()

        iCount += 1

    feat.Destroy()

    feat = src_soundg.GetNextFeature()

#############################################################################
# Cleanup


print(iCount, "features extracted")

csv_ds.Destroy()
ds.Destroy()

# Georgii's modification
import pandas as pd
data = pd.read_csv(csvfilename, header=0, index_col=None)
n_data = pd.DataFrame(data=data['LAT'].values, columns=['LAT'])
n_data['LONG'] = data['LONG'].values
n_data['DEPTH FT'] = data['DEPTHF'].values * -1
if len(csvfilename.split('.')) == 2:
    n_data.to_csv(csvfilename.split('.')[0] + '.xyz', index=None)
else:
    n_data.to_csv(csvfilename.split('.')[1].split('\\')[1] + '.xyz', index=None)

# KML
# Specify the target cCSV file in the same directory
filename, file_extension = os.path.splitext(s57filename)
csvfilename = filename
csvfilename += '.kml'

# Open the S57 file, and find the SOUNDG layer
ds = ogr.Open(s57filename)

try:
    src_soundg = ds.GetLayerByName('SOUNDG')
except AttributeError:
    print('')
    print('Error: SOUNDG layer or source S-57 file not found.')
    sys.exit(1)

# Create the output CSV file
csv_driver = ogr.GetDriverByName('KML')
csv_ds = csv_driver.CreateDataSource(csvfilename)

# Import CSR WGS84 (EPSG 4326)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

# Specify the new SOUNDG layer
csv_layer = csv_ds.CreateLayer('kml_soundg', srs, geom_type=ogr.wkbPoint25D)

src_defn = src_soundg.GetLayerDefn()
field_count = src_defn.GetFieldCount()

# Display progress message
print('')
print(csvfilename, "created ...")

# Copy the SOUNDG schema, duplicating the current fields, and create the LAT,
# LONG and DEPTH fields for the extracted Point and calculated data

out_mapping = []
for fld_index in range(field_count):

    src_fd = src_defn.GetFieldDefn(fld_index)
    fd = ogr.FieldDefn(src_fd.GetName(), src_fd.GetType())
    fd.SetWidth(src_fd.GetWidth())
    fd.SetPrecision(src_fd.GetPrecision())
    if csv_layer.CreateField(fd) != 0:
        out_mapping.append(-1)
    else:
        out_mapping.append(csv_layer.GetLayerDefn().GetFieldCount() - 1)

fd = ogr.FieldDefn('LAT', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATD', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATS', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATH', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LATDMS', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONG', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGD', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGS', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGH', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGDMS', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('DEPTHM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(0)
csv_layer.CreateField(fd)

fd = ogr.FieldDefn('DEPTHF', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(0)
csv_layer.CreateField(fd)

# Display progress message
print("SOUNDG layer created ...")

#  Process the data for the new fields
feat = src_soundg.GetNextFeature()

# The degree symbol needs to be defined for use
deg = u"\u00b0"

# GetFeatureCount can be expensive, so ...
iCount = 0

while feat is not None:

    multi_geom = feat.GetGeometryRef()

    for iPnt in range(multi_geom.GetGeometryCount()):
        # Get the multipart data
        pnt = multi_geom.GetGeometryRef(iPnt)

        # Get the layer definition from the KML file
        feat1 = ogr.Feature(feature_def=csv_layer.GetLayerDefn())

        for fld_index in range(field_count):
            feat1.SetField(out_mapping[fld_index], feat.GetField(fld_index))

        # Get the LATITUDE in Degrees.Decimal format
        soundinglat = pnt.GetX(0)

        # Extract the Degrees
        latdegrees = int(abs(soundinglat))
        # Extract the decimals
        trunclatdegrees = (abs(soundinglat) - latdegrees)
        # Calculate the Minutes
        latminutes = int(trunclatdegrees * 60)
        # Calculate the Seconds
        latseconds = round(((trunclatdegrees * 3600) % 60), 2)

        # Negative LATITUDE is West of the Zero Meridian
        if soundinglat <= 0:
            lathemisphere = "W"
        else:
            lathemisphere = "E"

        # Assemble the LATDMS string
        latstring = str(latdegrees)
        # Encode the defined degree symbol as utf-8 and add it
        latstring += str(deg.encode('utf8'))
        latstring += " "
        latstring += str(latminutes)
        latstring += "' "
        latstring += str(latseconds)
        latstring += '" '
        latstring += lathemisphere

        # Write the LATITUDE fields
        feat1.SetField('LAT', soundinglat)
        feat1.SetField('LATD', latdegrees)
        feat1.SetField('LATM', latminutes)
        feat1.SetField('LATS', latseconds)
        feat1.SetField('LATH', lathemisphere)
        feat1.SetField('LATDMS', latstring)

        # Get the LONGITUDE in Degrees.Decimal format
        soundinglong = pnt.GetY(0)

        # Extract the Degrees
        longdegrees = int(abs(soundinglong))
        # Extract the decimals
        trunclongdegrees = (abs(soundinglong) - longdegrees)
        # Calculate the Minutes
        longminutes = int(trunclongdegrees * 60)
        # Calculate the Seconds
        longseconds = round(((trunclongdegrees * 3600) % 60), 2)

        # Negative LONGITUDE is South of the equator
        if soundinglong <= 0:
            longhemisphere = "S"
        else:
            longhemisphere = "N"

        # Assemble the LONGDMS string
        longstring = str(longdegrees)
        # Encode the defined degree symbol as utf-8 and add it
        longstring += str(deg.encode('utf8'))
        longstring += " "
        longstring += str(longminutes)
        longstring += "' "
        longstring += str(longseconds)
        longstring += '" '
        longstring += longhemisphere

        # Write the LONGITUDE fields
        feat1.SetField('LONG', soundinglong)
        feat1.SetField('LONGD', longdegrees)
        feat1.SetField('LONGM', longminutes)
        feat1.SetField('LONGS', longseconds)
        feat1.SetField('LONGH', longhemisphere)
        feat1.SetField('LONGDMS', longstring)

        # Use metres as provided and calculate feet
        soundingmetres = pnt.GetZ(0)
        soundingfeet = soundingmetres * 3.28084

        # Write the DEPTH fields
        feat1.SetField('DEPTHM', soundingmetres)
        feat1.SetField('DEPTHF', soundingfeet)

        # Create the feature
        feat1.SetGeometry(pnt)
        csv_layer.CreateFeature(feat1)
        feat1.Destroy()

        iCount += 1

    feat.Destroy()

    feat = src_soundg.GetNextFeature()

#############################################################################
# Cleanup


print(iCount, "features extracted")

csv_ds.Destroy()
ds.Destroy()

