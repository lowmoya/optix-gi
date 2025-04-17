CC=g++
NVCC=nvcc
LIB_FOLDERS=./libraries/bin /opt/cuda/lib/
LIBS=nvoptix cudart assimp OpenEXR Imath
PATH:=/opt/cuda/bin:$(PATH)

SRC=source/
INC=include/
BIN=build/
DEV=device/
EXE=./optix_gi
DEP=dependencies/

INCLUDES=./libraries/include/ $(INC) /opt/cuda/include/ /usr/include/Imath

SRCOUT=$(BIN)$(SRC)
DEVOUT=$(BIN)$(DEV)
SRCDEPOUT=$(BIN)$(SRC)
DEVDEPOUT=$(BIN)$(DEV)

# Generative definitions
SRCFILES=$(shell find $(SRC) -name "*.cpp")
OBJFILES=$(SRCFILES:$(SRC)%.cpp=$(SRCOUT)%.o)
OBJFILES=$(patsubst $(SRC)%.cpp, $(SRCOUT)%.o, $(SRCFILES))
SRCDEPFILES=$(SRCFILES:$(SRC)%.cpp=$(SRCDEPOUT)%.d)
CUFILES=$(shell find $(DEV) -name "*.cu")
PTXFILES=$(CUFILES:$(DEV)%.cu=$(DEVOUT)%.ptx)
CUDEPFILES=$(CUFILES:$(DEV)%.cu=$(DEVDEPOUT)%.d)

LF=$(LIBS:%=-l%)
LDF=$(LIB_FOLDERS:%=-L%)
IF=$(INCLUDES:%=-I%)


# Named tasks
.PHONY: test clean

# ifneq ($(MAKECMDGOALS),clean)
# -include $(DEPFILES)
# endif
test: $(EXE)
	@echo -e '\n=> Starting Application'
	$(EXE)
clean:
	@echo -e '\n=> Removing Generated Content'
	rm -rf $(BIN) $(EXE) *.png


# File tasks
$(EXE):  $(PTXFILES) $(OBJFILES)
	@rm -f .compmark .depmark .ptxmark
	@echo -e '\n=> Linking Executable'
	$(CC) $(OBJFILES) -o $@ $(LDF) $(LF)

# Generative tasks
$(SRCDEPOUT)%.d: $(SRC)%.cpp
	@if [ ! -f .depmark ]; then \
		echo -e '\n=> Collecting Dependencies'; \
		rm -f .compmark; \
		touch .depmark; \
	fi
	@mkdir -p $(dir $@)
	$(CC) $(IF) $< -MM -MT $(@:$(SRCDEPOUT)%.d=$(SRCOUT)%.o) -MF $@

$(SRCOUT)%.o: $(SRC)%.cpp $(SRCDEPOUT)%.d
	@if [ ! -f .compmark ]; then \
		echo -e '\n=> Compiling Sources'; \
		rm -f .depmark; \
		touch .compmark; \
	fi

	@mkdir -p $(dir $@)
	$(CC) $(IF) $< -c -o $@

$(DEVDEPOUT)%.d: $(DEV)%.cu
	@if [ ! -f .depmark ]; then \
		echo -e '=> Collecting Dependencies'; \
		rm -f .compmark; \
		touch .depmark; \
	fi

	@mkdir -p $(dir $@)
	$(NVCC) $(IF) $< -MM -MT $(@:$(DEVDEPOUT)%.d=$(DEVOUT)%.ptx) -MF $@ -Wno-deprecated-gpu-targets

$(DEVOUT)%.ptx: $(DEV)%.cu $(DEVDEPOUT)%.d
	@if [ ! -f .ptxmark ]; then \
		echo  -e '\n=> Compiling PTX Files'; \
		rm -f .depmark; \
		touch .ptxmark; \
	fi

	@mkdir -p $(dir $@)
	$(NVCC) -ptx $(IF) $< -o $@ -Wno-deprecated-gpu-targets

-include  $(CUDEPFILES) $(SRCDEPFILES)
