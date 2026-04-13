//go:build linux
// +build linux

package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"image"
	"os"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// ENVIMapInfo holds georeferencing parameters parsed from the "map info" field.
type ENVIMapInfo struct {
	TiePixelX  float64 // reference pixel column (1-based)
	TiePixelY  float64 // reference pixel row (1-based)
	Easting    float64 // UTM easting of tie pixel (meters)
	Northing   float64 // UTM northing of tie pixel (meters)
	PixelSizeX float64 // pixel width in meters
	PixelSizeY float64 // pixel height in meters
	Zone       int     // UTM zone number
	North      bool    // true = North hemisphere
}

// ENVIHeader holds the relevant fields parsed from a .hdr file.
type ENVIHeader struct {
	Samples      int
	Lines        int
	Bands        int
	DataType     int    // 12 = uint16, 4 = float32, 2 = int16, 1 = uint8
	Interleave   string // bsq, bil, bip
	ByteOrder    int    // 0 = little-endian, 1 = big-endian
	HeaderOffset int
	MapInfo      *ENVIMapInfo // nil if not present
}

// ParseENVIHeader reads and parses an ENVI .hdr file.
func ParseENVIHeader(hdrPath string) (*ENVIHeader, error) {
	f, err := os.Open(hdrPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	h := &ENVIHeader{Interleave: "bsq"}
	scanner := bufio.NewScanner(f)

	var mapInfoRaw string
	inMapInfo := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// 处理跨行的 map info = { ... }
		if inMapInfo {
			mapInfoRaw += " " + line
			if strings.Contains(line, "}") {
				inMapInfo = false
				h.MapInfo = parseMapInfo(mapInfoRaw)
			}
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(strings.ToLower(parts[0]))
		val := strings.TrimSpace(parts[1])
		switch key {
		case "samples":
			h.Samples, _ = strconv.Atoi(val)
		case "lines":
			h.Lines, _ = strconv.Atoi(val)
		case "bands":
			h.Bands, _ = strconv.Atoi(val)
		case "data type":
			h.DataType, _ = strconv.Atoi(val)
		case "interleave":
			h.Interleave = strings.ToLower(val)
		case "byte order":
			h.ByteOrder, _ = strconv.Atoi(val)
		case "header offset":
			h.HeaderOffset, _ = strconv.Atoi(val)
		case "map info":
			if strings.Contains(val, "}") {
				h.MapInfo = parseMapInfo(val)
			} else {
				mapInfoRaw = val
				inMapInfo = true
			}
		}
	}
	if h.Samples == 0 || h.Lines == 0 {
		return nil, fmt.Errorf("HDR 解析失败: samples=%d lines=%d", h.Samples, h.Lines)
	}
	return h, nil
}

// parseMapInfo parses the value of the "map info" HDR field.
// Expected format: {UTM, tieX, tieY, easting, northing, pixSizeX, pixSizeY, zone, hemisphere, datum}
func parseMapInfo(raw string) *ENVIMapInfo {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "{")
	raw = strings.TrimSuffix(raw, "}")
	fields := strings.Split(raw, ",")
	if len(fields) < 9 {
		return nil
	}
	trim := func(s string) string { return strings.TrimSpace(s) }
	atof := func(s string) float64 {
		v, _ := strconv.ParseFloat(trim(s), 64)
		return v
	}
	atoi := func(s string) int {
		v, _ := strconv.Atoi(trim(s))
		return v
	}
	return &ENVIMapInfo{
		TiePixelX:  atof(fields[1]),
		TiePixelY:  atof(fields[2]),
		Easting:    atof(fields[3]),
		Northing:   atof(fields[4]),
		PixelSizeX: atof(fields[5]),
		PixelSizeY: atof(fields[6]),
		Zone:       atoi(fields[7]),
		North:      strings.EqualFold(trim(fields[8]), "north"),
	}
}

// bytesPerPixel returns the number of bytes for one pixel value given ENVI data type.
func bytesPerSample(dataType int) int {
	switch dataType {
	case 1:
		return 1 // uint8
	case 2, 12:
		return 2 // int16 / uint16
	case 4:
		return 4 // float32
	case 5:
		return 8 // float64
	default:
		return 2
	}
}

// readUint16LE reads a uint16 from a byte slice (little-endian).
func readUint16(b []byte, bigEndian bool) uint16 {
	if bigEndian {
		return binary.BigEndian.Uint16(b)
	}
	return binary.LittleEndian.Uint16(b)
}

// ENVITileReader holds the open file and header for random-access tile reads.
// The data file is memory-mapped for fast random access using Linux mmap.
type ENVITileReader struct {
	hdr       *ENVIHeader
	f         *os.File
	data      []byte // memory-mapped view of the entire file
	bps       int    // bytes per sample
	bigEndian bool
	// reusable scratch buffers (avoid per-tile allocation)
	pixels []uint16
	hist   []int
	lut    []uint8
}

// NewENVITileReader opens a DAT file and memory-maps it for tiled reading.
func NewENVITileReader(datPath string, hdr *ENVIHeader) (*ENVITileReader, error) {
	f, err := os.Open(datPath)
	if err != nil {
		return nil, err
	}

	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	fileSize := fi.Size()

	// Use Linux mmap
	data, err := syscall.Mmap(int(f.Fd()), 0, int(fileSize), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("mmap failed: %w", err)
	}

	return &ENVITileReader{
		hdr:       hdr,
		f:         f,
		data:      data,
		bps:       bytesPerSample(hdr.DataType),
		bigEndian: hdr.ByteOrder == 1,
		pixels:    make([]uint16, 3*tileSize*tileSize),
		hist:      make([]int, 65536),
		lut:       make([]uint8, 65536),
	}, nil
}

func (r *ENVITileReader) Close() {
	if len(r.data) > 0 {
		syscall.Munmap(r.data)
		r.data = nil
	}
	r.f.Close()
}

// ReadTileRGB reads a rectangular tile [x0,y0,x0+tileW,y0+tileH] from the BSQ
// ENVI file and returns an *image.NRGBA with uint16 values linearly stretched
// to uint8 (2% linear stretch per band).
func (r *ENVITileReader) ReadTileRGB(x0, y0, tileW, tileH int) (*image.NRGBA, error) {
	hdr := r.hdr
	W, H := hdr.Samples, hdr.Lines
	bps := r.bps

	// Clamp tile to image bounds
	x1 := x0 + tileW
	y1 := y0 + tileH
	if x1 > W {
		x1 = W
	}
	if y1 > H {
		y1 = H
	}
	actualW := x1 - x0
	actualH := y1 - y0
	n := actualW * actualH

	// We use up to 3 bands; map to R,G,B
	nBands := hdr.Bands
	if nBands > 3 {
		nBands = 3
	}

	// BSQ layout: band0 is all lines*samples, then band1, etc.
	bandSize := int64(W) * int64(H) * int64(bps)
	headerOffset := int64(hdr.HeaderOffset)
	d := r.data

	// Reuse scratch buffers from reader to avoid per-tile allocation
	pixels := r.pixels[:nBands*n]
	hist := r.hist
	lut := r.lut

	for b := 0; b < nBands; b++ {
		bandStart := headerOffset + int64(b)*bandSize
		dst := pixels[b*n : (b+1)*n]
		if !r.bigEndian {
			// Little-endian: copy rows directly via unsafe reinterpretation
			for row := 0; row < actualH; row++ {
				rowOff := bandStart + int64(y0+row)*int64(W)*int64(bps) + int64(x0)*int64(bps)
				srcBytes := d[rowOff : rowOff+int64(actualW*bps)]
				dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[row*actualW])), actualW*2)
				copy(dstBytes, srcBytes)
			}
		} else {
			for row := 0; row < actualH; row++ {
				rowOff := bandStart + int64(y0+row)*int64(W)*int64(bps) + int64(x0)*int64(bps)
				src := d[rowOff : rowOff+int64(actualW*bps)]
				for col := 0; col < actualW; col++ {
					dst[row*actualW+col] = binary.BigEndian.Uint16(src[col*2:])
				}
			}
		}
	}

	// Per-band 2% linear stretch; build LUT then apply directly into img.Pix
	img := image.NewNRGBA(image.Rect(0, 0, actualW, actualH))
	pix := img.Pix

	for b := 0; b < nBands; b++ {
		band := pixels[b*n : (b+1)*n]

		// Clear and fill histogram
		for i := range hist {
			hist[i] = 0
		}
		for _, v := range band {
			hist[v]++
		}

		low2pct := int(float64(n) * 0.02)
		high2pct := int(float64(n) * 0.98)
		var lo, hi uint16
		cum := 0
		for i, c := range hist {
			cum += c
			if cum >= low2pct {
				lo = uint16(i)
				break
			}
		}
		cum = 0
		for i := 65535; i >= 0; i-- {
			cum += hist[i]
			if cum >= (n - high2pct) {
				hi = uint16(i)
				break
			}
		}
		if hi <= lo {
			hi = lo + 1
		}
		scale := 255.0 / float64(hi-lo)

		// Build LUT
		for i := range lut {
			v := uint16(i)
			if v <= lo {
				lut[i] = 0
			} else if v >= hi {
				lut[i] = 255
			} else {
				lut[i] = uint8(float64(v-lo) * scale)
			}
		}

		// Write directly into img.Pix: NRGBA layout is R,G,B,A per pixel
		// Band mapping: b=0→B(offset 2), b=1→G(offset 1), b=2→R(offset 0)
		chanOffset := [3]int{2, 1, 0}[b]
		for i, v := range band {
			pix[i*4+chanOffset] = lut[v]
			pix[i*4+3] = 255 // alpha
		}
	}

	return img, nil
}

// linearStretch performs a 2% linear stretch on uint16 data to uint8.
func linearStretch(data []uint16, n int) []uint8 {
	return linearStretchWithHist(data, n, make([]int, 65536))
}

// linearStretchWithHist is like linearStretch but reuses an existing hist buffer.
func linearStretchWithHist(data []uint16, n int, hist []int) []uint8 {
	if n == 0 {
		return nil
	}
	// Clear and reuse hist
	for i := range hist {
		hist[i] = 0
	}
	for _, v := range data {
		hist[v]++
	}
	low2pct := int(float64(n) * 0.02)
	high2pct := int(float64(n) * 0.98)
	var lo, hi uint16
	cum := 0
	for i, c := range hist {
		cum += c
		if cum >= low2pct {
			lo = uint16(i)
			break
		}
	}
	cum = 0
	for i := 65535; i >= 0; i-- {
		cum += hist[i]
		if cum >= (n - high2pct) {
			hi = uint16(i)
			break
		}
	}
	if hi <= lo {
		hi = lo + 1
	}
	out := make([]uint8, n)
	scale := 255.0 / float64(hi-lo)
	for i, v := range data {
		if v <= lo {
			out[i] = 0
		} else if v >= hi {
			out[i] = 255
		} else {
			out[i] = uint8(float64(v-lo) * scale)
		}
	}
	return out
}
