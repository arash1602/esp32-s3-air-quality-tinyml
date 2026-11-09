idf_component_register(SRCS "main.c" "../../c/aq_infer.c" "../../bench/aq_bench.c"
                       INCLUDE_DIRS "../../c" "../../bench"
                       PRIV_REQUIRES esp_partition esp_timer driver)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
