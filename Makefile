CC=gcc
FLAGS=-Wall -fPIC

all:

clean:
	- rm *.so yolov3.h5
	- rm tests/cache/*
	- find . -iname __pycache__ -or -iname .pytest_cache | xargs rm -rf

test: helper.so
	python -m pytest tests

helper.so: helper.c libdarknet.so
	$(CC) $(CFLAGS) -shared $^ -o $@ libdarknet.so

libdarknet.so:
	ln -fs darknet/libdarknet.so .

archive:
	gnutar cjvf `date "+%Y%m%d"`.tar.bz2 --exclude-vcs-ignores --exclude yolov3.weights --exclude .git --exclude *.tar .
