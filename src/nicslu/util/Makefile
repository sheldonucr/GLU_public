include ../make.inc
COMPILE = $(CC) $(CFLAGS)
MAKELIB = $(AR)
INC = -I ../include

all: nicslu_util.a

OBJS = nicslu_readfile.o

nicslu_util.a: $(OBJS)
	$(MAKELIB) $@ $^
	$(RANLIB) $@

%.o: %.c
	$(COMPILE) $(INC) $< -o $@

clean:
	rm -f *.o nicslu_util.a