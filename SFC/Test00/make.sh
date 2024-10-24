
# Check if the "build" directory exists
if [ ! -d "build" ]; then
  # If not, create the directory
  echo "Creating build directory..."
  mkdir build
fi

# Enter the "build" directory
echo "Entering build directory..."
cd build

# Optional: Display the current directory
echo "Current directory: $(pwd)"

cmake ../../.. -DMNN_BUILD_SFC=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug -DMNN_AVX512=OFF

make -j20
