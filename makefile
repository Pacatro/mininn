# Makefile for install HDF5 dependencies and build the project

ALL:
	sudo apt update && sudo apt install -y libhdf5-dev
	cargo build