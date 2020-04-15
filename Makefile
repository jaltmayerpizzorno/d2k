CC=gcc
FLAGS=-Wall -fPIC

all:


clean:
	- rm *.so yolov3.h5
	- rm tests/cache/*
	- find . -iname __pycache__ -or -iname .pytest_cache | xargs rm -rf

test: helper.so
	python -m pytest tests

# we link with libdarknet.a (rather than "-L../darknet -ldarknet")
# to avoid dependencies when loading this DLL from Python
helper.so: helper.c
	$(CC) $(CFLAGS) -shared $^ -o $@ ../darknet/libdarknet.a

archive:
	gnutar cjvf `date "+%Y%m%d"`.tar.bz2 --exclude-vcs-ignores --exclude yolov3.weights --exclude .git --exclude *.tar .
