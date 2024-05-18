CC=clang
CFLAGS=-Wall -Wextra
LIBS=-lm -lraylib -pthread

build: main.c
	$(CC) $(CFLAGS) -o main main.c $(LIBS)

run: build
	./main
