##
## Shamelessly copied from craigsapp/midifile.
##

##############################
##
## Operating-system specific settings:
##

ARCH   =
ENV    =
OSTYPE =
ifeq ($(shell uname),Darwin)
   # This is an Apple OS X computer.
   OSTYPE = OSX

   # The MACOSX_DEPLOYMENT_TARGET allows you to compile on newer OS X computers,
   # but allows the code to run on older OS X computers.  In this case the
   # minimum OS version target will be 10.6:
   ENV = MACOSX_DEPLOYMENT_TARGET=10.9

   # The ARCH setting below forces the library to be compiled for
   # 32-bit architectures in OS X.  Uncomment the ARCH variable below
   # if compiling on a 64-bit computer, but you want 32-bit version
   # (for linking with other 32-bit libraries).
   #ARCH = -m32 -arch i386
else
   # This is probably a linux computer.
   OSTYPE = LINUX

   # The ARCH variable has to be set up slightly different for 32-bit compiling:
   #ARCH = -m32
endif

# Cygwin (and MinGW?) adds the string ".exe" to the end of compiled programs.
# so set EXTEN = .exe when compiling in cygwin. (Need a test for cygwin
# so that the EXTEN variable is setup automatically...)
EXTEN     =
# EXTEN   = .exe


##############################
#
# User-modifiable configuration variables:
#

SRCDIR    = src
INCDIR    = src/*/
TARGDIR   = bin
COMPILER  = LANG=C $(ENV) g++ $(ARCH)
DEFINES   = 
PREFLAGS  = -O3 -Wall $(DEFINES)
# Add -static flag to compile without dynamics libraries for better portability:
#PREFLAGS += -static

# Using C++ 2011 standard:
PREFLAGS += -std=c++11 -I$(INCDIR)
POSTFLAGS =

# MinGW compiling setup (used to compile for Microsoft Windows but actual
# compiling can be done in Linux). You have to install MinGW and these
# variables will probably have to be changed to the correct paths:
#COMPILER  = /opt/xmingw/bin/i386-mingw32msvc-g++
#TARGDIR   = bin-win
#POSTFLAGS = -Wl,--export-all-symbols -Wl,--enable-auto-import \
#            -Wl,--no-whole-archive -lmingw32

#                                                                         #
# End of user-modifiable variables.                                       #
#                                                                         #
###########################################################################


##############################
##
## Targets:
##

# Targets which don't actually refer to files
.PHONY:info clean


info:
	@echo ""
	@echo All directories inside ./$(SRCDIR)/ are treated as libraries, and thus are linked automatically.
	@echo All files inside ./$(SRCDIR)/ that are not directories are treated as standalone programs,
	@echo and are ignored unless set explicitly as build target.
	@echo Running this makefile will compile such specific program with all known libraries. Run:
	@echo "  make <program_name>"
	@echo For example, to build ./$(SRCDIR)/main.cpp with libraries in ./$(SRCDIR)/lib/ type:
	@echo "  make main"
	@echo Executables will be placed in ./$(TARGDIR)/.
	@echo ""


clean:
	@echo [*] Cleaning executables directory...
	@-rm -rf bin


%:
ifeq ($(wildcard $(TARGDIR)),)
	@-mkdir -p $(TARGDIR)
endif
	@echo [*] CC: Compiling $@...
	@$(COMPILER) $(PREFLAGS) $(SRCDIR)/*/*.cpp $(SRCDIR)/$@.cpp -o $(TARGDIR)/$@$(EXTEN) $< $(POSTFLAGS)
	@echo [*] CC: Done.


