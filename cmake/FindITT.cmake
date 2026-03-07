# FindITT.cmake — locate Intel ITT (Instrumentation and Tracing Technology) SDK
#
# Searches standard Intel oneAPI installation paths on Windows and Linux.
# Sets:
#   ITT_FOUND        — TRUE if both header and library found
#   ITT_INCLUDE_DIR  — path containing ittnotify.h
#   ITT_LIBRARY      — full path to libittnotify static library
#   ITT::ittnotify   — imported target (preferred usage)
#
# Tested against Intel oneAPI VTune 2023.2 / Advisor 2023.2 / Inspector 2024.1

cmake_minimum_required(VERSION 3.20)

# ── Candidate search roots ────────────────────────────────────────────────────

set(_ITT_WIN_ROOTS
    "C:/Program Files (x86)/Intel/oneAPI/vtune/2023.2.0/sdk"
    "C:/Program Files (x86)/Intel/oneAPI/vtune/latest/sdk"
    "C:/Program Files (x86)/Intel/oneAPI/advisor/2023.2.0/sdk"
    "C:/Program Files (x86)/Intel/oneAPI/advisor/latest/sdk"
    "C:/Program Files (x86)/Intel/oneAPI/inspector/2024.1/sdk"
    "C:/Program Files (x86)/Intel/oneAPI/inspector/latest/sdk"
    "C:/Program Files/Intel/oneAPI/vtune/latest/sdk"
)

set(_ITT_LINUX_ROOTS
    "/opt/intel/oneapi/vtune/latest/sdk"
    "/opt/intel/oneapi/advisor/latest/sdk"
    "/opt/intel/vtune_amplifier/sdk"
)

# ── Find header ───────────────────────────────────────────────────────────────

find_path(ITT_INCLUDE_DIR
    NAMES ittnotify.h
    HINTS
        ${_ITT_WIN_ROOTS}
        ${_ITT_LINUX_ROOTS}
    PATH_SUFFIXES include
    DOC "Intel ITT SDK include directory"
)

# ── Find library ──────────────────────────────────────────────────────────────

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_ITT_ARCH_SUFFIX "64")
else()
    set(_ITT_ARCH_SUFFIX "32")
endif()

if(WIN32)
    set(_ITT_LIB_NAME "libittnotify")
else()
    set(_ITT_LIB_NAME "ittnotify")
endif()

find_library(ITT_LIBRARY
    NAMES ${_ITT_LIB_NAME}
    HINTS
        ${_ITT_WIN_ROOTS}
        ${_ITT_LINUX_ROOTS}
    PATH_SUFFIXES
        "lib${_ITT_ARCH_SUFFIX}"
        "lib"
    DOC "Intel ITT SDK library"
)

# ── Standard result handling ──────────────────────────────────────────────────

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ITT
    REQUIRED_VARS ITT_INCLUDE_DIR ITT_LIBRARY
)

# ── Imported target ───────────────────────────────────────────────────────────

if(ITT_FOUND AND NOT TARGET ITT::ittnotify)
    add_library(ITT::ittnotify STATIC IMPORTED)
    set_target_properties(ITT::ittnotify PROPERTIES
        IMPORTED_LOCATION             "${ITT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ITT_INCLUDE_DIR}"
    )
    # ITT needs threading support on Linux
    if(UNIX)
        find_package(Threads REQUIRED)
        set_target_properties(ITT::ittnotify PROPERTIES
            INTERFACE_LINK_LIBRARIES Threads::Threads
        )
    endif()
endif()

mark_as_advanced(ITT_INCLUDE_DIR ITT_LIBRARY)
