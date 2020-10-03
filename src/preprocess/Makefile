include ../nicslu/make.inc
COMPILE = $(CC) $(CFLAGS)
INC = -I../nicslu/include -I../nicslu/util

preprocess.o: preprocess.c
	$(COMPILE) $(INC) $< -o $@

clean:
	rm -f preprocess.o
