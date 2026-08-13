cmake_minimum_required(VERSION 3.16)
project(fm_cssd_lda_firmware C)

add_library(fm_cssd_lda STATIC
    src/fm_cssd_lda.c
    src/fm_cssd_lda_params.c
)

target_include_directories(fm_cssd_lda
    PUBLIC include
    PRIVATE src
)
target_compile_features(fm_cssd_lda PUBLIC c_std_99)

if(NOT MSVC)
    target_link_libraries(fm_cssd_lda PUBLIC m)
endif()
