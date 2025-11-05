cat > test.sh << 'EOF'
#!/bin/bash

if [ ! -f "wget-list" ] && [ ! -f "*.tar.*" ]; then
    echo "❌ Запусти скрипт в папке /mnt/lfs/sources/"
    echo "💡 Сначала выполни: cd /mnt/lfs/sources"
    exit 1
fi

if [ -z "$1" ]; then
    echo "📦 Доступные пакеты:"
    echo "===================="
    ls -1 *.tar.* | cat -n
    echo ""
    echo "🚀 Использование: ./lfs-helper.sh <номер-пакета-или-имя>"
    echo "Пример: ./lfs-helper.sh 1"
    echo "Пример: ./lfs-helper.sh binutils-2.42.tar.xz"
    exit 0
fi

if [[ "$1" =~ ^[0-9]+$ ]]; then
    PACKAGE=$(ls *.tar.* | sed -n "${1}p")
else
    PACKAGE="$1"
fi

if [ ! -f "$PACKAGE" ]; then
    echo "❌ Пакет '$PACKAGE' не найден!"
    echo "📦 Доступные пакеты:"
    ls *.tar.*
    exit 1
fi

echo "ЗАПУСК СБОРКИ: $PACKAGE"
echo "========================================"

echo "Распаковываем..."
tar xvf "$PACKAGE"

FOLDER_NAME=$(echo "$PACKAGE" | sed 's/\.tar\..*//')
cd "$FOLDER_NAME"

echo ""
echo "УСПЕХ! Мы папке: $(pwd)"
echo "========================================"
echo ""
echo "ВЫПОЛНИ СЛЕДУЮЩИЕ КОМАНДЫ:"
echo ""

case "$PACKAGE" in
    binutils*)
        echo "BINUTILS:"
        echo "   mkdir -v build"
        echo "   cd build"
        echo "   ../configure --prefix=\$LFS/tools \\"
        echo "     --with-sysroot=\$LFS \\"
        echo "     --target=\$LFS_TGT \\"
        echo "     --disable-nls \\"
        echo "     --enable-gprofng=no \\"
        echo "     --disable-werror"
        echo "   make"
        echo "   make install"
        ;;
    gcc*)
        echo "GCC:"
        echo "   Требуются зависимости: mpfr, gmp, mpc"
        echo "   Если их нет - скачай сначала!"
        echo ""
        echo "   tar -xf ../mpfr-*.tar.*"
        echo "   tar -xf ../gmp-*.tar.*"
        echo "   tar -xf ../mpc-*.tar.*"
        echo "   mv -v mpfr-* mpfr"
        echo "   mv -v gmp-* gmp" 
        echo "   mv -v mpc-* mpc"
        echo ""
        echo "   mkdir -v build"
        echo "   cd build"
        echo "   ../configure --prefix=\$LFS/tools \\"
        echo "     --target=\$LFS_TGT \\"
        echo "     --disable-nls \\"
        echo "     --enable-languages=c,c++ \\"
        echo "     --disable-multilib \\"
        echo "     --disable-threads \\"
        echo "     --disable-libatomic \\"
        echo "     --disable-libgomp \\"
        echo "     --disable-libquadmath \\"
        echo "     --disable-libssp \\"
        echo "     --disable-libvtv \\"
        echo "     --disable-libstdcxx \\"
        echo "     --enable-default-pie \\"
        echo "     --enable-default-ssp"
        echo "   make"
        echo "   make install"
        ;;
    linux-*)
        echo "LINUX HEADERS:"
        echo "   make mrproper"
        echo "   make headers"
        echo "   find usr/include -name '.*' -delete"
        echo "   rm -f usr/include/Makefile"
        echo "   cp -rv usr/include \$LFS/usr"
        ;;
    glibc*)
        echo "GLIBC:"
        echo "   mkdir -v build"
        echo "   cd build"
        echo "   ../configure --prefix=/usr \\"
        echo "     --host=\$LFS_TGT \\"
        echo "     --build=\$(../scripts/config.guess) \\"
        echo "     --enable-kernel=4.19 \\"
        echo "     --with-headers=\$LFS/usr/include \\"
        echo "     --disable-werror"
        echo "   make"
        echo "   make DESTDIR=\$LFS install"
        ;;
    *)
        echo "Смотри LFS:"
        echo "   https://www.linuxfromscratch.org/lfs/view/stable-systemd/chapter05.html"
        echo ""
        echo "💡 Общие шаги:"
        echo "   1. mkdir build && cd build"
        echo "   2. ../configure --prefix=\$LFS/tools ..."
        echo "   3. make"
        echo "   4. make install"
        ;;
esac

echo ""
echo "========================================"
echo "⚡ После выполнения команд вернись в sources:"
echo "   cd /mnt/lfs/sources"
echo "💡 Следующий пакет: ./lfs-helper.sh <номер>"
EOF
