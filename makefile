# Makefile for Orion-X3/Orion-X4/mx and derivatives
# Written in 2011
# This makefile is licensed under the WTFPL


WARNINGS        = -Wno-padded -Wno-cast-align -Wno-unreachable-code -Wno-packed -Wno-missing-noreturn -Wno-float-equal -Wno-unused-macros -Werror=return-type -Wextra -Wno-unused-parameter -Wno-trigraphs -Wno-unused-local-typedef -Wno-reorder

COMMON_CFLAGS   = -Wall -O0 -g -march=native

CFLAGS          = $(COMMON_CFLAGS) -std=c99 -fPIC -O3
CXXFLAGS        = $(COMMON_CFLAGS) -Wno-old-style-cast -std=c++17 -ferror-limit=0 -fno-exceptions

CXXSRC          = $(shell find source -iname "*.cpp" -print)
CXXOBJ          = $(CXXSRC:.cpp=.cpp.o)
CXXDEPS         = $(CXXOBJ:.o=.d)

PRECOMP_HDRS    := source/include/precompile.h
PRECOMP_GCH     := $(PRECOMP_HDRS:.h=.h.gch)

DEFINES         = -DXTENSOR_ENABLE_CHECK_DIMENSION=1 -DXTENSOR_ENABLE_ASSERT=1
INCLUDES        = -Isource/include -Iexternal $(shell pkg-config --cflags xtensor)

.PHONY: all clean build
.PRECIOUS: $(PRECOMP_GCH)
.DEFAULT_GOAL = all

all: build
	@build/znn_test

build: build/znn_test

build/znn_test: $(CXXOBJ)
	@echo "  linking..."
	@$(CXX) $(CXXFLAGS) -o $@ $^

%.cpp.o: %.cpp makefile $(PRECOMP_GCH)
	@echo "  $(notdir $<)"
	@$(CXX) $(CXXFLAGS) $(WARNINGS) $(INCLUDES) $(DEFINES) -include source/include/precompile.h -MMD -MP -c -o $@ $<

%.c.o: %.c makefile
	@echo "  $(notdir $<)"
	@$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

%.h.gch: %.h makefile
	@printf "# precompiling header $<\n"
	@$(CXX) $(CXXFLAGS) $(WARNINGS) $(INCLUDES) -x c++-header -o $@ $<

clean:
	@find source -iname "*.cpp.d" | xargs rm
	@find source -iname "*.cpp.o" | xargs rm
	-@rm $(PRECOMP_GCH)

-include $(CXXDEPS)
-include $(CDEPS)












