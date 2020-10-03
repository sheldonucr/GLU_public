include ../make.inc
MAKELIB = $(AR)
COMPILE = $(CC) $(CFLAGS)

OBJS = nicslu_aegraph.o \
nicslu_amd.o \
nicslu_analyze.o \
nicslu_check.o \
nicslu_cond.o \
nicslu_destroy.o \
nicslu_dumpa.o \
nicslu_dumplu.o \
nicslu_etree.o \
nicslu_flops.o \
nicslu_global.o \
nicslu_initialize.o \
nicslu_internal.o \
nicslu_kernel.o \
nicslu_kernel_fact.o \
nicslu_kernel_fact_parallel.o \
nicslu_kernel_parallel.o \
nicslu_matrix.o \
nicslu_memuse.o \
nicslu_mergedup.o \
nicslu_mc64.o \
nicslu_mc64scale.o \
nicslu_refact.o \
nicslu_refact_parallel.o \
nicslu_refine.o \
nicslu_residual.o \
nicslu_scale.o \
nicslu_scheduler.o \
nicslu_solve.o \
nicslu_sort.o \
nicslu_static_symbolic.o \
nicslu_thread.o \
nicslu_thread_proc.o \
nicslu_throughput.o \
nicslu_transpose.o \
system.o \
thread.o \
timer_c.o

all: nicslu.a

nicslu.a: ${OBJS}
	$(MAKELIB) $@ $^
	$(RANLIB) $@

%.o: ../source/%.c
	$(COMPILE) $< -o $@

clean:
	rm -f *.o nicslu.a
