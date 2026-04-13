//go:build linux
// +build linux

package main

import (
	"os"
	"path/filepath"
)

// addThirdPartyToDllSearch sets ./third_party as an additional library search path on Linux.
// Uses LD_LIBRARY_PATH environment variable.
func addThirdPartyToDllSearch() error {
	absPath, err := filepath.Abs("./third_party")
	if err != nil {
		return err
	}

	// Add to LD_LIBRARY_PATH
	currentPath := os.Getenv("LD_LIBRARY_PATH")
	if currentPath == "" {
		os.Setenv("LD_LIBRARY_PATH", absPath)
	} else {
		os.Setenv("LD_LIBRARY_PATH", absPath+":"+currentPath)
	}

	return nil
}
