include ../make.inc
LIBS = -L. ../lib/nicslu.a ../util/nicslu_util.a -lrt -lpthread -lm
INC = -I ../include/ -I ../util/

all: demos demop test demos_time demop_time

test: test.o
	$(CC) $(LFLAGS) -o test test.o $(LIBS)

test.o: test.c
	$(CC) $(CFLAGS) -o test.o $(INC) test.c

demos_time: demos_time.o
	$(CC) $(LFLAGS) -o demos_time demos_time.o $(LIBS)

demos_time.o: demos_time.c
	$(CC) $(CFLAGS) -o demos_time.o $(INC) demos_time.c

demop_time: demop_time.o
	$(CC) $(LFLAGS) -o demop_time demop_time.o $(LIBS)

demop_time.o: demop_time.c
	$(CC) $(CFLAGS) -o demop_time.o $(INC) demop_time.c

demos: demos.o
	$(CC) $(LFLAGS) -o demos demos.o $(LIBS)

demos.o: demos.c
	$(CC) $(CFLAGS) -o demos.o $(INC) demos.c

demop: demop.o
	$(CC) $(LFLAGS) -o demop demop.o $(LIBS)

demop.o: demop.c
	$(CC) $(CFLAGS) -o demop.o $(INC) demop.c

clean:
	rm -f demos.o demos demop.o demop test.o test demos_time demos_time.o demop_time demop_time.o
