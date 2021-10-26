cmake_minimum_required(VERSION 2.8.12)

project(midifile C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##############################
##
## Operating-system specific settings:
##

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

    option(STATIC_CRT "Use static CRT libraries" OFF)

    # Rewrite command line flags to use /MT if necessary
    if(STATIC_CRT)
        foreach(flag_var
                CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            if(${flag_var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endif(${flag_var} MATCHES "/MD")
        endforeach(flag_var)
    endif()
endif()


##############################
##
## Library:
##

set(SRCS
    src/Options.cpp
    src/Binasc.cpp
    src/MidiEvent.cpp
    src/MidiEventList.cpp
    src/MidiFile.cpp
    src/MidiMessage.cpp
)

set(HDRS
    include/midifile/Binasc.h
    include/midifile/MidiEvent.h
    include/midifile/MidiEventList.h
    include/midifile/MidiFile.h
    include/midifile/MidiMessage.h
    include/midifile/Options.h
)

add_library(midifile STATIC ${SRCS} ${HDRS})

target_include_directories(midifile PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/midifile> $<INSTALL_INTERFACE:include/midifile>)
target_include_directories(midifile PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
