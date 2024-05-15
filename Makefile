CFLAGS=-Wall -Wextra -std=c11 -pedantic
LIBS=-lm

build: main.c
	$(CC) $(CFLAGS) -o main main.c $(LIBS)
