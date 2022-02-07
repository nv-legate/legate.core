# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ifndef LIBNAME
$(error LIBNAME must be given, aborting build)
endif

USE_PGI ?= 0
# Check to see if this is the PGI compiler
# in whch case we need to use different flags in some cases
ifeq ($(strip $(USE_PGI)),0)
ifeq ($(findstring nvc++,$(shell $(CXX) --version)),nvc++)
USE_PGI = 1
endif
endif
# Check to see if we are building on Mac OS
ifeq ($(shell uname -s),Darwin)
DARWIN = 1
ifeq ($(strip $(USE_OPENMP)),1)
$(warning "Some versions of Clang on Mac OSX do not support OpenMP")
endif
endif

RM	:= rm

CC_FLAGS ?=
CC_FLAGS += -std=c++14 -Wfatal-errors
CC_FLAGS += -I$(LEGATE_DIR)/include

ifneq ($(strip $(BOOTSTRAP)), 1)
# Pull in the default values for the configuration
include $(LEGATE_DIR)/share/legate/config.mk
LD_FLAGS += -llgcore
endif
LD_FLAGS += -L$(LEGATE_DIR)/lib -llegion -lrealm -Wl,-rpath,$(LEGATE_DIR)/lib

ifeq ($(strip $(USE_CUDA)),1)
ifeq (,$(shell which nvcc))
  NVCC ?= $(CUDA)/bin/nvcc
else
  NVCC ?= $(shell which nvcc)
endif
endif

NVCC_FLAGS ?=
NVCC_FLAGS += -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -ccbin=$(CXX)
NVCC_FLAGS += -I$(LEGATE_DIR)/include

ifeq ($(strip $(DEBUG)),1)
ifeq ($(strip $(DARWIN)),1)
CC_FLAGS   += -O0 -glldb
else ifeq ($(strip $(USE_PGI)),1)
CC_FLAGS   += -O0 -g
else
CC_FLAGS   += -O0 -ggdb #-ggdb -Wall
endif
NVCC_FLAGS += -g -O0
else
CC_FLAGS   += -O2 -fno-strict-aliasing #-ggdb
NVCC_FLAGS += -O2
endif

ifeq ($(strip $(DEBUG_RELEASE)),1)
ifeq ($(strip $(DARWIN)),1)
CC_FLAGS   += -glldb
else ifeq ($(strip $(USE_PGI)),1)
CC_FLAGS   += -g
else
CC_FLAGS   += -ggdb #-ggdb -Wall
endif
NVCC_FLAGS += -g
endif

ifeq ($(strip $(USE_CUDA)),1)
# translate legacy arch names into numbers
ifeq ($(strip $(GPU_ARCH)),fermi)
override GPU_ARCH = 20
NVCC_FLAGS	+= -DFERMI_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),kepler)
override GPU_ARCH = 30
NVCC_FLAGS	+= -DKEPLER_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),k20)
override GPU_ARCH = 35
NVCC_FLAGS	+= -DK20_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),k80)
override GPU_ARCH = 37
NVCC_FLAGS	+= -DK80_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),maxwell)
override GPU_ARCH = 52
NVCC_FLAGS	+= -DMAXWELL_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),pascal)
override GPU_ARCH = 60
NVCC_FLAGS	+= -DPASCAL_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),volta)
override GPU_ARCH = 70
NVCC_FLAGS	+= -DVOLTA_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),turing)
override GPU_ARCH = 75
NVCC_FLAGS	+= -DTURING_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),ampere)
override GPU_ARCH = 80
NVCC_FLAGS	+= -DAMPERE_ARCH
endif

COMMA=,
NVCC_FLAGS += $(foreach X,$(subst $(COMMA), ,$(GPU_ARCH)),-gencode arch=compute_$(X)$(COMMA)code=sm_$(X))
CC_FLAGS	+= -DLEGATE_USE_CUDA -I$(CUDA)/include
NVCC_FLAGS	+= -DLEGATE_USE_CUDA -I$(CUDA)/include
LD_FLAGS	+= -L$(CUDA)/lib -L$(CUDA)/lib64
endif

GEN_SRC		?=
GEN_CPU_SRC	?=
GEN_CPU_SRC	+= $(GEN_SRC)

GEN_CPU_DEPS	:= $(GEN_CPU_SRC:.cc=.cc.d)
GEN_CPU_OBJS	:= $(GEN_CPU_SRC:.cc=.cc.o)
ifeq ($(strip $(USE_CUDA)),1)
GEN_GPU_DEPS	:= $(GEN_GPU_SRC:.cu=.cu.d)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.cu.o)
else
GEN_GPU_DEPS	:=
GEN_GPU_OBJS	:=
endif

CC_FLAGS += -fPIC
NVCC_FLAGS += --compiler-options '-fPIC'
ifeq ($(shell uname), Darwin)
	DLIB = $(LIBNAME).dylib
	LD_FLAGS += -dynamiclib -fPIC -install_name @rpath/$(DLIB)
else
	DLIB = $(LIBNAME).so
	LD_FLAGS += -shared
endif

OMP_FLAGS ?=
ifeq ($(strip $(USE_OPENMP)),1)
OMP_FLAGS 	+= -fopenmp
CC_FLAGS 	+= -DLEGATE_USE_OPENMP
endif

ifeq ($(strip $(USE_GASNET)),1)
CC_FLAGS	+= -DLEGATE_USE_GASNET
endif

.PHONY: all
all: $(DLIB)

.PHONY: install
ifdef PREFIX
INSTALL_HEADERS ?=
install: $(PREFIX)/lib/$(DLIB) $(addprefix $(PREFIX)/include/,$(INSTALL_HEADERS))
$(PREFIX)/include/%.h: %.h
	mkdir -p $(dir $@)
	cp $< $@
$(PREFIX)/include/%.inl: %.inl
	mkdir -p $(dir $@)
	cp $< $@
$(PREFIX)/lib/$(DLIB): $(DLIB)
	mkdir -p $(dir $@)
	cp $< $@
ifeq ($(strip $(BOOTSTRAP)), 1)
install: $(PREFIX)/share/legate/legate.mk # in addition to items above
$(PREFIX)/share/legate/legate.mk: legate.mk
	mkdir -p $(dir $@)
	cp $< $@
endif
else
install:
	$(error Must specify PREFIX for installation)
endif

$(DLIB) : $(GEN_CPU_OBJS) $(GEN_GPU_OBJS)
	@echo "---> Linking objects into one library: $(DLIB)"
	$(CXX) -o $(DLIB) $^ $(LD_FLAGS)

-include $(GEN_CPU_DEPS)

$(GEN_CPU_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(INC_FLAGS) $(OMP_FLAGS) $(CC_FLAGS)

-include $(GEN_GPU_DEPS)

$(GEN_GPU_OBJS) : %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(NVCC) -o $<.d -M -MT $@ $< $(INC_FLAGS) $(NVCC_FLAGS)
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	$(RM) -f $(DLIB) $(GEN_CPU_DEPS) $(GEN_CPU_OBJS) $(GEN_GPU_DEPS) $(GEN_GPU_OBJS)

# disable gmake's default rule for building % from %.o
% : %.o
