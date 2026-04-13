package main

import "math"

// utmToLatLon converts UTM coordinates to WGS-84 latitude/longitude (degrees).
// zone: UTM zone number, north: true for northern hemisphere.
func utmToLatLon(easting, northing float64, zone int, north bool) (lat, lon float64) {
	// WGS-84 ellipsoid parameters
	const (
		a  = 6378137.0 // semi-major axis
		f  = 1.0 / 298.257223563
		k0 = 0.9996   // scale factor
		E0 = 500000.0 // false easting
	)
	N0 := 0.0
	if !north {
		N0 = 10000000.0 // false northing for southern hemisphere
	}

	b := a * (1 - f)
	e2 := 1 - (b*b)/(a*a)
	e1 := (1 - math.Sqrt(1-e2)) / (1 + math.Sqrt(1-e2))

	x := easting - E0
	y := northing - N0

	M := y / k0
	mu := M / (a * (1 - e2/4 - 3*e2*e2/64 - 5*e2*e2*e2/256))

	phi1 := mu +
		(3*e1/2-27*e1*e1*e1/32)*math.Sin(2*mu) +
		(21*e1*e1/16-55*e1*e1*e1*e1/32)*math.Sin(4*mu) +
		(151*e1*e1*e1/96)*math.Sin(6*mu) +
		(1097*e1*e1*e1*e1/512)*math.Sin(8*mu)

	N1 := a / math.Sqrt(1-e2*math.Sin(phi1)*math.Sin(phi1))
	T1 := math.Tan(phi1) * math.Tan(phi1)
	C1 := e2 / (1 - e2) * math.Cos(phi1) * math.Cos(phi1)
	R1 := a * (1 - e2) / math.Pow(1-e2*math.Sin(phi1)*math.Sin(phi1), 1.5)
	D := x / (N1 * k0)

	lat = phi1 - (N1*math.Tan(phi1)/R1)*(D*D/2-
		(5+3*T1+10*C1-4*C1*C1-9*e2/(1-e2))*D*D*D*D/24+
		(61+90*T1+298*C1+45*T1*T1-252*e2/(1-e2)-3*C1*C1)*D*D*D*D*D*D/720)

	lon0 := float64((zone-1)*6-180+3) * math.Pi / 180
	lon = lon0 + (D-
		(1+2*T1+C1)*D*D*D/6+
		(5-2*C1+28*T1-3*C1*C1+8*e2/(1-e2)+24*T1*T1)*D*D*D*D*D/120)/math.Cos(phi1)

	lat = lat * 180 / math.Pi
	lon = lon * 180 / math.Pi
	return
}

// pixelToLatLon converts a pixel coordinate (col, row) in the full image
// to WGS-84 latitude/longitude using the ENVI map info.
// col and row are 0-based.
func pixelToLatLon(col, row float64, mi *ENVIMapInfo) (lat, lon float64) {
	// ENVI tie pixel is 1-based
	easting := mi.Easting + (col-(mi.TiePixelX-1))*mi.PixelSizeX
	northing := mi.Northing - (row-(mi.TiePixelY-1))*mi.PixelSizeY
	return utmToLatLon(easting, northing, mi.Zone, mi.North)
}
