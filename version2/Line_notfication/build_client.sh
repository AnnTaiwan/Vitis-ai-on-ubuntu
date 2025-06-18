CXX=${CXX:-g++} # if CXX exists value, then use CXX. If not, then use g++
$CXX -O3 client_final.cpp -o client_final -lcurl -ljson-c -pthread
# by using
# export CXX="aarch64-xilinx-linux-g++  -mcpu=cortex-a72.cortex-a53 -march=armv8-a+crc -fstack-protector-strong  -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -Werror=format-security --sysroot=$SDKTARGETSYSROOT"
