# Makefile for install HDF5 dependencies and build the project

all:
	@set -e; \
	if command -v apt-get >/dev/null 2>&1; then \
		echo "Using apt-get..."; \
		sudo apt update && sudo apt install -y libhdf5-dev; \
	elif command -v dnf >/dev/null 2>&1; then \
		echo "Using dnf..."; \
		sudo dnf install -y hdf5-devel; \
	elif command -v yum >/dev/null 2>&1; then \
		echo "Using yum..."; \
		sudo yum install -y hdf5-devel; \
	elif command -v pacman >/dev/null 2>&1; then \
		echo "Using pacman..."; \
		sudo pacman -Syu --noconfirm hdf5; \
	elif command -v zypper >/dev/null 2>&1; then \
		echo "Using zypper..."; \
		sudo zypper install -y hdf5-devel; \
	elif command -v apk >/dev/null 2>&1; then \
		echo "Using apk (Alpine Linux)..."; \
		sudo apk add --no-cache hdf5-dev; \
	else \
		echo "No supported package manager found. Please install HDF5 dependencies manually."; \
		exit 1; \
	fi; \
	cargo build

