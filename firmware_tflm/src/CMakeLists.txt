idf_component_register(SRCS "main.cc"
                       INCLUDE_DIRS "." "../../bench"
                       PRIV_REQUIRES esp_partition esp_timer esp_driver_usb_serial_jtag esp_driver_gpio)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-variable)
