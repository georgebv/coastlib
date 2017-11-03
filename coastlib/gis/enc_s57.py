###############################################################################
# $Id: s57_kml_extract_soundg.py 2017-06-27 Dave Liske <dave@micuisine.com>
#
# Purpose: Extract SOUNDGings from an S-57 dataset, and write them to
# KML format, creating one feature for each sounding, and
# adding the latitudes, longitudes and depths as attributes
# for easier use.
# Authors: Frank Warmerdam, 2003 version, warmerdam@pobox.com
# Dave Liske, 2017 version, dave@micuisine.com
#
###############################################################################
# Copyright (c) 2003, Frank Warmerdam <warmerdam@pobox.com>
# Additional Copyright (c) 2017, Dave Liske <dave@micuisine.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
###############################################################################-

try:
    from osgeo import ogr
except ImportError:
    import ogr

import sys
import os.path
import osgeo.osr as osr


###############################################################################-
def Usage():
    print('')
    print('Usage: s57_kml_extract_soundg.py ')
    sys.exit(1)


def NoSoundG():
    print('')
    print('Error: SOUNDG layer or source S-57 file not found.')
    sys.exit(1)


###############################################################################-
# Argument processing

if len(sys.argv) != 2:
    Usage()

s57filename = sys.argv[1]

# Specify the target KML file in the same directory
filename, file_extension = os.path.splitext(s57filename)
kmlfilename = filename
kmlfilename += ".kml"

#############################################################################-
# Open the S57 file, and find the SOUNDG layer.

ds = ogr.Open(s57filename)

try:
    src_soundg = ds.GetLayerByName('SOUNDG')
except AttributeError:
    NoSoundG()

#############################################################################-
# Create the output KML file.

kml_driver = ogr.GetDriverByName('KML')
kml_ds = kml_driver.CreateDataSource(kmlfilename)

# Import CSR WGS84
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

# Specify the new SOUNDG layer
kml_layer = kml_ds.CreateLayer('kml_soundg', srs, geom_type=ogr.wkbPoint25D)

src_defn = src_soundg.GetLayerDefn()
field_count = src_defn.GetFieldCount()

# Display progress message
print('')
print(kmlfilename, "created ...")

#############################################################################-
# Copy the SOUNDG schema, duplicating the current fields, and create the LAT,
#	LONG and DEPTH fields for the extracted Point and calculated data

out_mapping = []
for fld_index in range(field_count):

    src_fd = src_defn.GetFieldDefn(fld_index)
    fd = ogr.FieldDefn(src_fd.GetName(), src_fd.GetType())
    fd.SetWidth(src_fd.GetWidth())
    fd.SetPrecision(src_fd.GetPrecision())
    if kml_layer.CreateField(fd) != 0:
        out_mapping.append(-1)
    else:
        out_mapping.append(kml_layer.GetLayerDefn().GetFieldCount() - 1)

fd = ogr.FieldDefn('LAT', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LATD', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LATM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LATS', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LATH', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LATDMS', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LONG', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGD', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGS', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGH', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('LONGDMS', ogr.OFTString)
fd.SetWidth(12)
fd.SetPrecision(7)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('DEPTHM', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(0)
kml_layer.CreateField(fd)

fd = ogr.FieldDefn('DEPTHF', ogr.OFTReal)
fd.SetWidth(12)
fd.SetPrecision(0)
kml_layer.CreateField(fd)

# Display progress message
print("SOUNDG layer created ...")

#############################################################################
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
        feat1 = ogr.Feature(feature_def=kml_layer.GetLayerDefn())

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
        kml_layer.CreateFeature(feat1)
        feat1.Destroy()

        iCount += 1

    feat.Destroy()

    feat = src_soundg.GetNextFeature()

#############################################################################
# Cleanup

print(iCount, "features extracted")

kml_ds.Destroy()
ds.Destroy()
