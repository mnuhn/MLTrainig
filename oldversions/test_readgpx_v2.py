import gpxpy

gpx_file = open('Wednesday_Classic.gpx')
gpx = gpxpy.parse(gpx_file)

lat = []
lon = []

for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            lat.append(point.latitude)
            lon.append(point.longitude)

print lat

