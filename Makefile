CC=g++
NVCC=nvcc
LIB_FOLDERS=./libraries/bin
LIBS=nvoptix cudart

SRC=source/
INC=include/
BIN=build/
DEV=device/
EXE=./optix_gi
DEP=dependencies/

INCLUDES=./libraries/include $(INC)

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
	@echo "\n>- Starting Application"
	$(EXE)
clean:
	@echo "\n>- Removing Generated Content"
	rm -rf $(BIN) $(EXE) output.png


# File tasks
$(EXE):  $(PTXFILES) $(OBJFILES)
	@rm -f .compmark
	@echo "\n>- Linking Executable"
	@echo $^
	$(CC) $(OBJFILES) -o $@ $(LDF) $(LF)

# Generative tasks
$(SRCDEPOUT)%.d: $(SRC)%.cpp
	@if [ ! -f .depmark ]; then \
		echo "\n>- Collecting Dependencies"; \
		rm -f .compmark; \
		touch .depmark; \
	fi
	@mkdir -p $(dir $@)
	$(CC) $(IF) $< -MM -MT $(@:$(SRCDEPOUT)%.d=$(SRCOUT)%.o) -MF $@

$(SRCOUT)%.o: $(SRC)%.cpp $(SRCDEPOUT)%.d
	@if [ ! -f .compmark ]; then \
		echo "\n>- Compiling Sources"; \
		rm -f .depmark; \
		touch .compmark; \
	fi

	@mkdir -p $(dir $@)
	$(CC) $(IF) $< -c -o $@

$(DEVDEPOUT)%.d: $(DEV)%.cu
	@if [ ! -f .depmark ]; then \
		echo "\n>- Collecting Dependencies"; \
		rm -f .compmark; \
		touch .depmark; \
	fi

	@mkdir -p $(dir $@)
	$(NVCC) $(IF) $< -MM -MT $(@:$(DEVDEPOUT)%.d=$(DEVOUT)%.ptx) -MF $@

$(DEVOUT)%.ptx: $(DEV)%.cu $(DEVDEPOUT)%.d
	@if [ ! -f .compmark ]; then \
		echo "\n>- Compiling PTX Files"; \
		rm -f .depmark; \
		touch .compmark; \
	fi

	@mkdir -p $(dir $@)
	$(NVCC) -ptx $(IF) $< -o $@

-include  $(CUDEPFILES) $(SRCDEPFILES)