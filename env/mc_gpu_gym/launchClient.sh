#!/bin/bash

replaceable=0
port=0
seed="NONE"
maxMem="12G"
device="egl"
fatjar=build/libs/mcprec-6.13.jar
MyCustomMap=/mnt/shared-storage-user/luozhihao/simluater_test/engine/templete/new

while [ $# -gt 0 ]
do
    case "$1" in
        -replaceable) replaceable=1;;
        -port) port="$2"; shift;;
        -seed) seed="$2"; shift;;
        -maxMem) maxMem="$2"; shift;;
        -device) device="$2"; shift;;
        -fatjar) fatjar="$2"; shift;;
        -working_dir) working_dir="$2"; shift;;
        *) echo >&2 \
            "usage: $0 [-replaceable] [-port <port>] [-seed <seed>] [-maxMem <maxMem>] [-device <device>] [-fatjar <fatjar>] [-working_dir <working_dir>]"
            exit 1;;
    esac
    shift
done

#!/bin/bash
#!/bin/bash -e

# 注释掉危险的操作：rm -rf /tmp/.X* 会删除所有 X11 socket，导致多实例冲突
# rm -rf /tmp/.X*
# 只清理当前 DISPLAY 的 socket（如果需要）
if [ -n "$port" ] && [ "$port" != "0" ]; then
    rm -f /tmp/.X${port}-lock /tmp/.X11-unix/X${port} 2>/dev/null || true
fi
export PATH="${PATH}:/opt/VirtualGL/bin"
export LD_LIBRARY_PATH="/usr/lib/libreoffice/program:${LD_LIBRARY_PATH}"

# /etc/init.d/dbus start
export DISPLAY=":$port"
echo "fffffffffffffffffffffffffffffffffffffffffffff"
echo "${DISPLAY}"
Xvfb "${DISPLAY}" -ac -screen "0" "1920x1200x24" -dpi "72" +extension "RANDR" +extension "GLX" +iglx +extension "MIT-SHM" +render -nolisten "tcp" -noreset -shmem -maxclients 128 &

# x11vnc -display :${DISPLAY} -forever -shared -nopw -rfbport 5900 &

# Wait for X11 to start
echo "Waiting for X socket"
until [ -S "/tmp/.X11-unix/X${DISPLAY/:/}" ]; do sleep 1; done
echo "X socket is ready"


echo "Session Running."

"$@"


# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# 设置 NVIDIA 环境变量，但不使用 VirtualGL
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# 禁用 MineStudio 的 VirtualGL 集成，使用原生渲染
export MINESTUDIO_GPU_RENDER=1
export VGL_DISPLAY=""
export VGL_REFRESHRATE="60"
export PATH="${PATH}:/opt/VirtualGL/bin"

export PYTHONPATH="/mnt/shared-storage-user/luozhihao/envs/jarvis_vla:${PYTHONPATH}"

export INST_NAME=1219
export INST_ID=1219

export MC_ROOT=$working_dir
  
export INST_DIR=$MC_ROOT/versions/1219
export INST_MC_DIR=$MC_ROOT/versions/1219
export INST_JAVA=/mnt/shared-storage-user/steai_share/zhanglechao/miniconda3/envs/mcvla/bin/java
export INST_NEOFORGE=1

if ! [[ $port =~ ^-?[0-9]+$ ]]; then
    echo "Port value should be numeric"
    exit 1
fi


if [ \( $port -lt 0 \) -o \( $port -gt 65535 \) ]; then
    echo "Port value out of range 0-65535"
    exit 1
fi


export MAP_NAME="tmeo"


until xdpyinfo -display ${DISPLAY} >/dev/null 2>&1; do sleep 0.1; done

cd $MC_ROOT/versions/1219
vglrun -d $device /mnt/shared-storage-user/steai_share/zhanglechao/miniconda3/envs/mcvla/bin/java -Xmx13610m "-Dfile.encoding=UTF-8" "-Dstdout.encoding=UTF-8" "-Dstderr.encoding=UTF-8" "-Djava.rmi.server.useCodebaseOnly=true" "-Dcom.sun.jndi.rmi.object.trustURLCodebase=false" "-Dcom.sun.jndi.cosnaming.object.trustURLCodebase=false" "-Dlog4j2.formatMsgNoLookups=true" "-Dlog4j.configurationFile=$MC_ROOT/versions/1219/log4j2.xml" "-Dminecraft.client.jar=.minecraft/versions/1219/1219.jar" "-Duser.home=null" -XX:+UnlockExperimentalVMOptions -XX:+UseG1GC "-XX:G1NewSizePercent=20" "-XX:G1ReservePercent=20" "-XX:MaxGCPauseMillis=50" "-XX:G1HeapRegionSize=32m" -XX:-UseAdaptiveSizePolicy -XX:-OmitStackTraceInFastThrow -XX:-DontCompileHugeMethods "-Dfml.ignoreInvalidMinecraftCertificates=true" "-Dfml.ignorePatchDiscrepancies=true" "-Djava.library.path=$MC_ROOT/versions/1219/natives-linux-x86_64" "-Djna.tmpdir=$MC_ROOT/versions/1219/natives-linux-x86_64" "-Dorg.lwjgl.system.SharedLibraryExtractPath=$MC_ROOT/versions/1219/natives-linux-x86_64" "-Dio.netty.native.workdir=$MC_ROOT/versions/1219/natives-linux-x86_64" "-Dminecraft.launcher.brand=HMCL" "-Dminecraft.launcher.version=3.6.12" -cp $MC_ROOT/libraries/net/neoforged/fancymodloader/earlydisplay/4.0.39/earlydisplay-4.0.39.jar:$MC_ROOT/libraries/net/neoforged/fancymodloader/loader/4.0.39/loader-4.0.39.jar:$MC_ROOT/libraries/net/neoforged/accesstransformers/at-modlauncher/10.0.1/at-modlauncher-10.0.1.jar:$MC_ROOT/libraries/net/neoforged/accesstransformers/10.0.1/accesstransformers-10.0.1.jar:$MC_ROOT/libraries/net/neoforged/bus/8.0.2/bus-8.0.2.jar:$MC_ROOT/libraries/net/neoforged/coremods/7.0.3/coremods-7.0.3.jar:$MC_ROOT/libraries/cpw/mods/modlauncher/11.0.4/modlauncher-11.0.4.jar:$MC_ROOT/libraries/net/neoforged/mergetool/2.0.0/mergetool-2.0.0-api.jar:$MC_ROOT/libraries/com/electronwill/night-config/toml/3.8.2/toml-3.8.2.jar:$MC_ROOT/libraries/com/electronwill/night-config/core/3.8.2/core-3.8.2.jar:$MC_ROOT/libraries/net/neoforged/JarJarSelector/0.4.1/JarJarSelector-0.4.1.jar:$MC_ROOT/libraries/net/neoforged/JarJarMetadata/0.4.1/JarJarMetadata-0.4.1.jar:$MC_ROOT/libraries/org/apache/maven/maven-artifact/3.8.5/maven-artifact-3.8.5.jar:$MC_ROOT/libraries/net/jodah/typetools/0.6.3/typetools-0.6.3.jar:$MC_ROOT/libraries/net/minecrell/terminalconsoleappender/1.3.0/terminalconsoleappender-1.3.0.jar:$MC_ROOT/libraries/net/fabricmc/sponge-mixin/0.15.2+mixin.0.8.7/sponge-mixin-0.15.2+mixin.0.8.7.jar:$MC_ROOT/libraries/org/openjdk/nashorn/nashorn-core/15.4/nashorn-core-15.4.jar:$MC_ROOT/libraries/org/apache/commons/commons-lang3/3.14.0/commons-lang3-3.14.0.jar:$MC_ROOT/libraries/cpw/mods/bootstraplauncher/2.0.2/bootstraplauncher-2.0.2.jar:$MC_ROOT/libraries/cpw/mods/securejarhandler/3.0.8/securejarhandler-3.0.8.jar:$MC_ROOT/libraries/org/ow2/asm/asm-commons/9.7/asm-commons-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm-util/9.7/asm-util-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm-analysis/9.7/asm-analysis-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm-tree/9.7/asm-tree-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm/9.7/asm-9.7.jar:$MC_ROOT/libraries/net/neoforged/JarJarFileSystems/0.4.1/JarJarFileSystems-0.4.1.jar:$MC_ROOT/libraries/net/sf/jopt-simple/jopt-simple/5.0.4/jopt-simple-5.0.4.jar:$MC_ROOT/libraries/org/slf4j/slf4j-api/2.0.9/slf4j-api-2.0.9.jar:$MC_ROOT/libraries/org/antlr/antlr4-runtime/4.13.1/antlr4-runtime-4.13.1.jar:$MC_ROOT/libraries/com/mojang/logging/1.2.7/logging-1.2.7.jar:$MC_ROOT/libraries/org/apache/logging/log4j/log4j-slf4j2-impl/2.22.1/log4j-slf4j2-impl-2.22.1.jar:$MC_ROOT/libraries/org/apache/logging/log4j/log4j-core/2.22.1/log4j-core-2.22.1.jar:$MC_ROOT/libraries/org/apache/logging/log4j/log4j-api/2.22.1/log4j-api-2.22.1.jar:$MC_ROOT/libraries/org/jline/jline-reader/3.20.0/jline-reader-3.20.0.jar:$MC_ROOT/libraries/org/jline/jline-terminal/3.20.0/jline-terminal-3.20.0.jar:$MC_ROOT/libraries/commons-io/commons-io/2.15.1/commons-io-2.15.1.jar:$MC_ROOT/libraries/net/minecraftforge/srgutils/0.4.15/srgutils-0.4.15.jar:$MC_ROOT/libraries/com/google/guava/guava/32.1.2-jre/guava-32.1.2-jre.jar:$MC_ROOT/libraries/com/google/guava/failureaccess/1.0.1/failureaccess-1.0.1.jar:$MC_ROOT/libraries/com/google/guava/listenablefuture/9999.0-empty-to-avoid-conflict-with-guava/listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar:$MC_ROOT/libraries/com/google/code/findbugs/jsr305/3.0.2/jsr305-3.0.2.jar:$MC_ROOT/libraries/org/checkerframework/checker-qual/3.33.0/checker-qual-3.33.0.jar:$MC_ROOT/libraries/com/google/errorprone/error_prone_annotations/2.18.0/error_prone_annotations-2.18.0.jar:$MC_ROOT/libraries/com/google/j2objc/j2objc-annotations/2.8/j2objc-annotations-2.8.jar:$MC_ROOT/libraries/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar:$MC_ROOT/libraries/org/codehaus/plexus/plexus-utils/3.3.0/plexus-utils-3.3.0.jar:$MC_ROOT/libraries/com/machinezoo/noexception/noexception/1.7.1/noexception-1.7.1.jar:$MC_ROOT/libraries/com/github/oshi/oshi-core/6.4.10/oshi-core-6.4.10.jar:$MC_ROOT/libraries/com/ibm/icu/icu4j/73.2/icu4j-73.2.jar:$MC_ROOT/libraries/com/mojang/authlib/6.0.54/authlib-6.0.54.jar:$MC_ROOT/libraries/com/mojang/blocklist/1.0.10/blocklist-1.0.10.jar:$MC_ROOT/libraries/com/mojang/brigadier/1.3.10/brigadier-1.3.10.jar:$MC_ROOT/libraries/com/mojang/datafixerupper/8.0.16/datafixerupper-8.0.16.jar:$MC_ROOT/libraries/com/mojang/patchy/2.2.10/patchy-2.2.10.jar:$MC_ROOT/libraries/com/mojang/text2speech/1.17.9/text2speech-1.17.9.jar:$MC_ROOT/libraries/commons-codec/commons-codec/1.16.0/commons-codec-1.16.0.jar:$MC_ROOT/libraries/commons-logging/commons-logging/1.2/commons-logging-1.2.jar:$MC_ROOT/libraries/io/netty/netty-buffer/4.1.97.Final/netty-buffer-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-codec/4.1.97.Final/netty-codec-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-common/4.1.97.Final/netty-common-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-handler/4.1.97.Final/netty-handler-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-resolver/4.1.97.Final/netty-resolver-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-transport-classes-epoll/4.1.97.Final/netty-transport-classes-epoll-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-transport-native-epoll/4.1.97.Final/netty-transport-native-epoll-4.1.97.Final-linux-aarch_64.jar:$MC_ROOT/libraries/io/netty/netty-transport-native-epoll/4.1.97.Final/netty-transport-native-epoll-4.1.97.Final-linux-x86_64.jar:$MC_ROOT/libraries/io/netty/netty-transport-native-unix-common/4.1.97.Final/netty-transport-native-unix-common-4.1.97.Final.jar:$MC_ROOT/libraries/io/netty/netty-transport/4.1.97.Final/netty-transport-4.1.97.Final.jar:$MC_ROOT/libraries/it/unimi/dsi/fastutil/8.5.12/fastutil-8.5.12.jar:$MC_ROOT/libraries/net/java/dev/jna/jna-platform/5.14.0/jna-platform-5.14.0.jar:$MC_ROOT/libraries/net/java/dev/jna/jna/5.14.0/jna-5.14.0.jar:$MC_ROOT/libraries/org/apache/commons/commons-compress/1.26.0/commons-compress-1.26.0.jar:$MC_ROOT/libraries/org/apache/httpcomponents/httpclient/4.5.13/httpclient-4.5.13.jar:$MC_ROOT/libraries/org/apache/httpcomponents/httpcore/4.4.16/httpcore-4.4.16.jar:$MC_ROOT/libraries/org/jcraft/jorbis/0.0.17/jorbis-0.0.17.jar:$MC_ROOT/libraries/org/joml/joml/1.10.5/joml-1.10.5.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-freetype/3.3.3/lwjgl-freetype-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-freetype/3.3.3/lwjgl-freetype-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-glfw/3.3.3/lwjgl-glfw-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-glfw/3.3.3/lwjgl-glfw-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-jemalloc/3.3.3/lwjgl-jemalloc-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-jemalloc/3.3.3/lwjgl-jemalloc-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-openal/3.3.3/lwjgl-openal-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-openal/3.3.3/lwjgl-openal-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-opengl/3.3.3/lwjgl-opengl-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-opengl/3.3.3/lwjgl-opengl-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-stb/3.3.3/lwjgl-stb-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-stb/3.3.3/lwjgl-stb-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-tinyfd/3.3.3/lwjgl-tinyfd-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl-tinyfd/3.3.3/lwjgl-tinyfd-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl/3.3.3/lwjgl-3.3.3.jar:$MC_ROOT/libraries/org/lwjgl/lwjgl/3.3.3/lwjgl-3.3.3-natives-linux.jar:$MC_ROOT/libraries/org/lz4/lz4-java/1.8.0/lz4-java-1.8.0.jar:$MC_ROOT/versions/1219/1219.jar "-Djava.net.preferIPv6Addresses=system" "-DignoreList=client-extra,1219.jar,1219.jar" "-DlibraryDirectory=$MC_ROOT/libraries" -p $MC_ROOT/libraries/cpw/mods/bootstraplauncher/2.0.2/bootstraplauncher-2.0.2.jar:$MC_ROOT/libraries/cpw/mods/securejarhandler/3.0.8/securejarhandler-3.0.8.jar:$MC_ROOT/libraries/org/ow2/asm/asm-commons/9.7/asm-commons-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm-util/9.7/asm-util-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm-analysis/9.7/asm-analysis-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm-tree/9.7/asm-tree-9.7.jar:$MC_ROOT/libraries/org/ow2/asm/asm/9.7/asm-9.7.jar:$MC_ROOT/libraries/net/neoforged/JarJarFileSystems/0.4.1/JarJarFileSystems-0.4.1.jar --add-modules ALL-MODULE-PATH --add-opens "java.base/java.util.jar=cpw.mods.securejarhandler" --add-opens "java.base/java.lang.invoke=cpw.mods.securejarhandler" --add-exports "java.base/sun.security.util=cpw.mods.securejarhandler" --add-exports "jdk.naming.dns/com.sun.jndi.dns=java.naming" cpw.mods.bootstraplauncher.BootstrapLauncher --username Nepenthes --version 1219 --gameDir $MC_ROOT/versions/1219 --assetsDir $MC_ROOT/assets --assetIndex 17 --uuid 9135db0a9ab6363dbddaa6a3e01e31e0 --accessToken c830816478ec43e5a931b457f35187d0 --clientId "\${clientid}" --xuid "\${auth_xuid}" --userType msa --versionType "HMCL 3.6.12" --width 854 --height 480 --fml.neoForgeVersion 21.1.168 --fml.fmlVersion 4.0.39 --fml.mcVersion 1.21.1 --fml.neoFormVersion 20240808.144430 --launchTarget forgeclient --quickPlaySingleplayer "$MAP_NAME"



[ $replaceable -gt 0 ]

